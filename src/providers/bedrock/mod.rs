use std::sync::Arc;
use std::time::SystemTime;

use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    SignableBody, SignableRequest, SigningSettings, sign as sign_http_request,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::providers::chat_params::{
    apply_chat_param_rules, merge_chat_params, resolve_chat_param_rules,
};
use crate::providers::common::{
    header_map_from_config, parse_response_json, reject_custom_tool_calls,
};
use crate::types::chat::ChatResponse;
use crate::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatStream, ProviderDriver,
    ProviderId, ProviderInit, ProviderKind, ProviderKindStatic, SigmaError, SigmaResult,
    StreamBehavior, submit_provider,
};

mod config;
mod request;
mod response;
mod stream;

use config::{
    BedrockAuth, BedrockConfig, resolve_api_base_override, resolve_auth, resolve_configured_region,
    resolve_region,
};
use request::{BedrockState, bedrock_request_body, build_provider_state, endpoint};
use response::{bedrock_error_response, bedrock_response_to_chat};
use stream::BedrockConverseStream;

const BEDROCK_KIND: ProviderKindStatic = ProviderKindStatic::new("bedrock");
const BEDROCK_DEFAULT_REGION: &str = "us-west-2";
const JSON_TOOL_NAME: &str = "json_tool_call";

/// Semantic chat parameters this adapter exposes through the Bedrock Converse API.
///
/// `parallel_tool_calls` and `stream_options` are intentionally absent: the
/// Bedrock Converse API does not expose either control, and silently dropping
/// them previously hid configuration mistakes. Callers that still need to
/// experiment with these fields can opt in via
/// [`crate::ChatParamConfig::allow`] or
/// [`crate::types::chat::ChatRequest::provider_options`].
const SUPPORTED_CHAT_PARAMS: &[&str] = &[
    "guardrailConfig",
    "max_completion_tokens",
    "max_tokens",
    "outputConfig",
    "performanceConfig",
    "reasoning_effort",
    "requestMetadata",
    "response_format",
    "service_tier",
    "stop",
    "stream",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "web_search",
];

struct BedrockProvider {
    id: ProviderId,
    kind: ProviderKind,
    chat: BedrockChatAdapter,
}

impl BedrockProvider {
    fn from_config(init: ProviderInit<BedrockConfig>) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let headers = header_map_from_config(&init.id, init.common.headers.clone())?;
        let auth = resolve_auth(&init, headers.contains_key(AUTHORIZATION))?;
        let region = resolve_configured_region(init.config.region.as_deref());
        let api_base = resolve_api_base_override(&init, init.config.runtime_endpoint.as_deref());

        Ok(Arc::new(Self {
            id: init.id.clone(),
            kind: init.kind,
            chat: BedrockChatAdapter {
                provider: init.id,
                api_base,
                region,
                auth,
                headers,
            },
        }))
    }
}

impl ProviderDriver for BedrockProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn kind(&self) -> &ProviderKind {
        &self.kind
    }

    fn chat(&self) -> Option<&dyn ChatCompletionAdapter> {
        Some(&self.chat)
    }
}

struct BedrockChatAdapter {
    provider: ProviderId,
    api_base: Option<String>,
    region: Option<String>,
    auth: BedrockAuth,
    headers: HeaderMap,
}

impl ChatCompletionAdapter for BedrockChatAdapter {
    fn endpoint(&self, request: &ChatAdapterRequest<'_>) -> SigmaResult<ProviderEndpoint> {
        let stream = request.streaming;
        let region = self.region_for_model(request.context.provider_model);
        let api_base = self.api_base_for_region(&region);
        Ok(endpoint(&api_base, request.context.provider_model, stream))
    }

    fn transform_request(
        &self,
        request: ChatAdapterRequest<'_>,
        endpoint: ProviderEndpoint,
    ) -> SigmaResult<ProviderRequest> {
        reject_custom_tool_calls(&self.provider, &request.request.messages)?;
        let region = self.region_for_model(request.context.provider_model);
        let inject_stream = request.streaming && self.stream_behavior().inject_stream;
        let mut params =
            merge_chat_params(request.deployment_defaults, request.request, inject_stream)?;
        let rules = resolve_chat_param_rules(
            SUPPORTED_CHAT_PARAMS,
            request.chat_param_config,
            request.context.provider_model,
        );
        apply_chat_param_rules(&self.provider, &mut params, &rules)?;

        let mut translated = bedrock_request_body(
            &self.provider,
            request.context.clone(),
            &request.request.messages,
            &params,
        )?;
        if let Some(provider_options) = request.request.provider_options.get(&self.provider) {
            translated.body.extend(provider_options.clone());
        }

        let mut headers = self.headers.clone();
        insert_header_if_missing(
            &mut headers,
            CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        insert_header_if_missing(
            &mut headers,
            ACCEPT,
            HeaderValue::from_static("application/json"),
        );

        Ok(ProviderRequest {
            method: endpoint.method,
            url: endpoint.url,
            headers,
            body: Value::Object(translated.body),
            provider_state: Some(Arc::new(build_provider_state(
                translated.reverse_tool_map,
                &region,
            )) as crate::ProviderState),
        })
    }

    fn sign_request(&self, mut request: ProviderRequest) -> SigmaResult<SignedProviderRequest> {
        match &self.auth {
            BedrockAuth::Header => {}
            BedrockAuth::Bearer(token) => {
                if !request.headers.contains_key(AUTHORIZATION) {
                    let value = format!("Bearer {}", token.expose_secret());
                    request.headers.insert(
                        AUTHORIZATION,
                        HeaderValue::from_str(&value).map_err(|err| {
                            SigmaError::ProviderSigning {
                                provider: self.provider.clone(),
                                message: err.to_string(),
                            }
                        })?,
                    );
                }
            }
            BedrockAuth::SigV4(credentials) => {
                // TODO: Add full AWS credential-chain support here: default provider chain,
                // shared config profiles, AssumeRole, web identity, and refreshable credentials
                // need async resolution or a provider lifecycle extension beyond this sync hook.
                let region = signing_region(&request).unwrap_or_else(|| {
                    self.region
                        .clone()
                        .unwrap_or_else(|| BEDROCK_DEFAULT_REGION.to_string())
                });
                self.sign_sigv4(&mut request, credentials, &region)?;
            }
        }

        Ok(request.into())
    }

    fn transform_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaResult<ChatResponse> {
        let body = parse_response_json(&self.provider, response.body.as_ref())?;
        bedrock_response_to_chat(context, body)
    }

    fn transform_error_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaError {
        bedrock_error_response(context, response)
    }

    fn transform_stream(
        &self,
        context: &ChatAdapterContext<'_>,
        stream: ProviderByteStream,
    ) -> SigmaResult<ChatStream> {
        Ok(Box::pin(BedrockConverseStream::new(
            self.provider.clone(),
            context.provider_model.to_string(),
            stream,
            request::reverse_tool_map(context),
        )))
    }

    fn stream_behavior(&self) -> StreamBehavior {
        StreamBehavior::native(true)
    }
}

impl BedrockChatAdapter {
    fn sign_sigv4(
        &self,
        request: &mut ProviderRequest,
        credentials: &config::AwsCredentials,
        region: &str,
    ) -> SigmaResult<()> {
        let body = request.body.to_string();
        let credentials = Credentials::new(
            credentials.access_key_id.expose_secret(),
            credentials.secret_access_key.expose_secret(),
            credentials
                .session_token
                .as_ref()
                .map(|token| token.expose_secret().to_string()),
            None,
            "sigma-bedrock-static",
        );
        let identity = Identity::from(credentials);
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(region)
            .name("bedrock")
            .time(SystemTime::now())
            .settings(SigningSettings::default())
            .build()
            .map_err(|err| SigmaError::ProviderSigning {
                provider: self.provider.clone(),
                message: err.to_string(),
            })?
            .into();
        let header_values = request
            .headers
            .iter()
            .map(|(name, value)| {
                value
                    .to_str()
                    .map(|value| (name.as_str().to_string(), value.to_string()))
                    .map_err(|err| SigmaError::ProviderSigning {
                        provider: self.provider.clone(),
                        message: format!("invalid header value for `{name}`: {err}"),
                    })
            })
            .collect::<SigmaResult<Vec<_>>>()?;
        let signable_request = SignableRequest::new(
            request.method.as_str(),
            request.url.as_str(),
            header_values
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
            SignableBody::Bytes(body.as_bytes()),
        )
        .map_err(|err| SigmaError::ProviderSigning {
            provider: self.provider.clone(),
            message: err.to_string(),
        })?;
        let (instructions, _signature) = sign_http_request(signable_request, &signing_params)
            .map_err(|err| SigmaError::ProviderSigning {
                provider: self.provider.clone(),
                message: err.to_string(),
            })?
            .into_parts();

        let (headers, params) = instructions.into_parts();
        if !params.is_empty() {
            return Err(SigmaError::ProviderSigning {
                provider: self.provider.clone(),
                message: "bedrock SigV4 signing unexpectedly returned query parameters".to_string(),
            });
        }
        for header in headers {
            let name = HeaderName::from_static(header.name());
            let mut value = HeaderValue::from_str(header.value()).map_err(|err| {
                SigmaError::ProviderSigning {
                    provider: self.provider.clone(),
                    message: format!("invalid signed header `{name}`: {err}"),
                }
            })?;
            value.set_sensitive(header.sensitive());
            request.headers.insert(name, value);
        }

        Ok(())
    }
}

impl BedrockChatAdapter {
    fn region_for_model(&self, model: &crate::ModelName) -> String {
        resolve_region(self.region.as_deref(), model)
    }

    fn api_base_for_region(&self, region: &str) -> String {
        self.api_base
            .clone()
            .unwrap_or_else(|| format!("https://bedrock-runtime.{region}.amazonaws.com"))
    }
}

fn signing_region(request: &ProviderRequest) -> Option<String> {
    request
        .provider_state
        .as_ref()
        .and_then(|state| state.downcast_ref::<BedrockState>())
        .map(|state| state.signing_region.clone())
}

fn insert_header_if_missing(headers: &mut HeaderMap, name: HeaderName, value: HeaderValue) {
    if !headers.contains_key(&name) {
        headers.insert(name, value);
    }
}

submit_provider! {
    kind: BEDROCK_KIND,
    constructor: BedrockProvider::from_config,
    config: BedrockConfig,
}
