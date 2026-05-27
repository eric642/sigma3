use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use futures_util::StreamExt;
use serde_json::Value;

use crate::config::{
    ChatParamConfig, ChatParamModelConfig, ChatParameterMap, ClientConfig, ParamPolicy,
};
use crate::provider::{
    ChatAdapterContext, ChatAdapterRequest, ChatStream, ProviderCatalog, ProviderInit, StreamMode,
    deployment_model_info, response_to_stream_chunk,
};
use crate::types::chat::{ChatRequest, ChatResponse};
use crate::{
    DeploymentId, ModelDeploymentConfig, ModelName, ModelRef, ProviderDriver, ProviderId,
    SigmaError, SigmaResult,
};

/// Entry point for calling configured LLM providers.
///
/// A client owns initialized provider instances, deployment routing tables, and
/// a shared [`reqwest::Client`]. Build one from [`ClientConfig`] and reuse it
/// across requests.
///
/// ```rust
/// # use sigma::{Client, ClientConfig};
/// # fn example() -> sigma::SigmaResult<()> {
/// let client = Client::build(ClientConfig::default())?;
/// let _chat = client.chat();
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    config: ClientConfig,
    providers: HashMap<ProviderId, Arc<dyn ProviderDriver>>,
    provider_chat_params: HashMap<ProviderId, ChatParamConfig>,
    deployments_by_id: HashMap<DeploymentId, ModelDeploymentConfig>,
    deployments_by_public_model: HashMap<ModelName, DeploymentId>,
    http_client: reqwest::Client,
}

/// Builder for runtime resources shared by a [`Client`].
///
/// Provider constructors come from the inventory registry. This builder only
/// accepts runtime resources, such as the HTTP client used for provider calls.
#[derive(Clone)]
pub struct ClientBuilder {
    http_client: reqwest::Client,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self {
            http_client: reqwest::Client::new(),
        }
    }
}

impl ClientBuilder {
    /// Installs the reqwest client used for provider HTTP calls.
    ///
    /// The same client is used for all provider instances. Applications can
    /// configure the [`reqwest::Client`] with their own timeouts, proxies, TLS,
    /// tracing middleware, or test server settings before passing it here.
    pub fn with_http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = http_client;
        self
    }

    /// Builds a client from runtime resources and [`ClientConfig`].
    ///
    /// # Errors
    ///
    /// Returns an error when provider registrations are duplicated, a provider
    /// kind is unknown, provider instance ids are duplicated, deployments are
    /// duplicated, or a deployment references an unknown provider instance.
    pub fn build(self, config: ClientConfig) -> SigmaResult<Client> {
        let catalog = ProviderCatalog::from_inventory()?;
        let mut providers = HashMap::new();
        let mut provider_chat_params = HashMap::new();

        for provider_config in &config.providers {
            if providers.contains_key(&provider_config.id) {
                return Err(SigmaError::DuplicateProviderInstance {
                    id: provider_config.id.clone(),
                });
            }

            let constructor = catalog.get(&provider_config.kind).ok_or_else(|| {
                SigmaError::UnknownProviderKind {
                    kind: provider_config.kind.clone(),
                }
            })?;
            let provider = constructor(ProviderInit::from(provider_config.clone()))?;
            provider_chat_params.insert(
                provider_config.id.clone(),
                provider_config.common.chat_params.clone(),
            );
            providers.insert(provider_config.id.clone(), provider);
        }

        let mut deployments_by_id = HashMap::new();
        let mut deployments_by_public_model = HashMap::new();

        for deployment in &config.deployments {
            if !providers.contains_key(&deployment.provider) {
                return Err(SigmaError::ProviderConfig {
                    provider: Some(deployment.provider.clone()),
                    message: format!(
                        "deployment `{}` references an unknown provider instance",
                        deployment.id
                    ),
                });
            }

            if deployments_by_id
                .insert(deployment.id.clone(), deployment.clone())
                .is_some()
            {
                return Err(SigmaError::DuplicateDeployment {
                    id: deployment.id.clone(),
                });
            }

            if deployments_by_public_model
                .insert(deployment.public_model.clone(), deployment.id.clone())
                .is_some()
            {
                return Err(SigmaError::ProviderConfig {
                    provider: Some(deployment.provider.clone()),
                    message: format!(
                        "multiple deployments expose public model `{}`",
                        deployment.public_model
                    ),
                });
            }
        }

        Ok(Client {
            inner: Arc::new(ClientInner {
                config,
                providers,
                provider_chat_params,
                deployments_by_id,
                deployments_by_public_model,
                http_client: self.http_client,
            }),
        })
    }
}

impl Client {
    /// Starts a client builder.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Builds a client with default runtime resources.
    ///
    /// This uses a default [`reqwest::Client`]. Applications that need custom
    /// HTTP runtime behavior should call [`Client::builder`] and configure it
    /// with [`ClientBuilder::with_http_client`].
    ///
    /// # Errors
    ///
    /// Returns the same configuration errors as [`ClientBuilder::build`].
    pub fn build(config: ClientConfig) -> SigmaResult<Self> {
        Self::builder().build(config)
    }

    /// Returns the chat completions API.
    ///
    /// Use the returned namespace to create regular or streaming chat
    /// completions:
    /// `client.chat().create(&request).await`.
    pub fn chat(&self) -> ChatNamespace<'_> {
        ChatNamespace { client: self }
    }
}

/// Chat completions API namespace.
///
/// Requests are routed through [`ClientConfig`] deployments before being sent
/// to a configured provider. Use [`ChatNamespace::create`] for a single
/// response or [`ChatNamespace::create_stream`] for streaming responses.
pub struct ChatNamespace<'a> {
    client: &'a Client,
}

impl ChatNamespace<'_> {
    /// Creates one chat completion.
    ///
    /// The call resolves [`ChatRequest::model`] through
    /// deployment routing, runs the provider adapter lifecycle, sends the signed
    /// request with the configured HTTP client, and transforms the provider
    /// response back into sigma's semantic chat response type.
    ///
    /// # Errors
    ///
    /// Returns routing, unsupported parameter, provider adapter, HTTP, or
    /// provider response errors.
    pub async fn create(
        &self,
        request: &ChatRequest,
    ) -> SigmaResult<crate::types::chat::ChatResponse> {
        self.client.create_chat_completion(request).await
    }

    /// Creates a streaming chat completion.
    ///
    /// The provider adapter's [`crate::StreamBehavior`] controls whether sigma
    /// injects `"stream": true`, uses the HTTP stream path, or produces a
    /// fake one-item stream from a non-streaming response.
    ///
    /// # Errors
    ///
    /// Returns routing, unsupported parameter, provider adapter, HTTP, or
    /// provider response errors.
    pub async fn create_stream(&self, request: &ChatRequest) -> SigmaResult<ChatStream> {
        self.client.create_chat_completion_stream(request).await
    }
}

#[derive(Clone)]
struct ResolvedRoute {
    provider: Arc<dyn ProviderDriver>,
    deployment: Option<ModelDeploymentConfig>,
    public_model: ModelName,
    provider_model: ModelName,
}

impl Client {
    async fn create_chat_completion(
        &self,
        request: &ChatRequest,
    ) -> SigmaResult<crate::types::chat::ChatResponse> {
        let route = self.resolve_route(&request.model)?;

        if let Some(custom_chat) = route.provider.custom_chat() {
            return custom_chat.create(route.to_routed_request(request)).await;
        }

        let provider = Arc::clone(&route.provider);
        let adapter = provider
            .chat()
            .ok_or_else(|| SigmaError::ProviderDoesNotSupportChat {
                provider: provider.id().clone(),
            })?;
        let (signed_request, adapter, context) =
            self.prepare_provider_request(request, &route, adapter, None)?;
        let response = self.execute_http(signed_request).await?;

        transform_response_or_error(adapter, &context, response)
    }

    async fn create_chat_completion_stream(
        &self,
        request: &ChatRequest,
    ) -> SigmaResult<ChatStream> {
        let route = self.resolve_route(&request.model)?;

        if let Some(custom_chat) = route.provider.custom_chat() {
            return custom_chat
                .create_stream(route.to_routed_request(request))
                .await;
        }

        let provider = Arc::clone(&route.provider);
        let adapter = provider
            .chat()
            .ok_or_else(|| SigmaError::ProviderDoesNotSupportChat {
                provider: provider.id().clone(),
            })?;
        let stream_behavior = adapter.stream_behavior();
        let (signed_request, adapter, context) =
            self.prepare_provider_request(request, &route, adapter, Some(stream_behavior))?;

        if stream_behavior.mode == StreamMode::FakeFromResponse {
            let response = self.execute_http(signed_request).await?;
            let response = transform_response_or_error(adapter, &context, response)?;
            return Ok(Box::pin(futures_util::stream::once(async move {
                Ok(response_to_stream_chunk(response))
            })));
        }

        let byte_stream = self.stream_http(signed_request, adapter, &context).await?;
        adapter.transform_stream(&context, byte_stream)
    }

    async fn execute_http(
        &self,
        request: crate::SignedProviderRequest,
    ) -> SigmaResult<crate::ProviderResponse> {
        let response = self
            .http_request(request)
            .send()
            .await
            .map_err(http_error)?;

        provider_response(response).await
    }

    async fn stream_http(
        &self,
        request: crate::SignedProviderRequest,
        adapter: &dyn crate::ChatCompletionAdapter,
        context: &crate::ChatAdapterContext<'_>,
    ) -> SigmaResult<crate::ProviderByteStream> {
        let response = self
            .http_request(request)
            .send()
            .await
            .map_err(http_error)?;
        let status = response.status();

        if !status.is_success() {
            let response = provider_response(response).await?;
            return Err(adapter.transform_error_response(context, response));
        }

        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(http_error));

        Ok(Box::pin(stream))
    }

    fn http_request(&self, request: crate::SignedProviderRequest) -> reqwest::RequestBuilder {
        self.inner
            .http_client
            .request(request.method, request.url)
            .headers(request.headers)
            .body(request.body.to_string())
    }

    fn prepare_provider_request<'a>(
        &self,
        request: &'a ChatRequest,
        route: &'a ResolvedRoute,
        adapter: &'a dyn crate::ChatCompletionAdapter,
        stream_behavior: Option<crate::StreamBehavior>,
    ) -> SigmaResult<(
        crate::SignedProviderRequest,
        &'a dyn crate::ChatCompletionAdapter,
        crate::ChatAdapterContext<'a>,
    )> {
        let mut params = self.chat_params(request, route.deployment.as_ref())?;
        if stream_behavior.is_some_and(|behavior| behavior.inject_stream) {
            params.insert("stream".to_string(), Value::Bool(true));
        }

        let context = route.adapter_context();
        let rules =
            self.resolve_chat_param_rules(context.provider, context.provider_model, adapter);
        let params = self.apply_chat_param_rules(context.provider, params, &rules)?;
        let params = adapter.map_chat_params(params)?;
        adapter.validate_environment()?;

        let provider_options = request.provider_options.get(context.provider);
        let adapter_request = ChatAdapterRequest {
            context: context.clone(),
            messages: &request.messages,
            params,
            provider_options,
        };

        let endpoint = adapter.endpoint(&adapter_request)?;
        let provider_request = adapter.transform_request(adapter_request, endpoint)?;
        let signed_request = adapter.sign_request(provider_request)?;
        let context = ChatAdapterContext {
            provider_state: signed_request.provider_state.clone(),
            ..context
        };

        Ok((signed_request, adapter, context))
    }

    fn chat_params(
        &self,
        request: &ChatRequest,
        deployment: Option<&ModelDeploymentConfig>,
    ) -> SigmaResult<ChatParameterMap> {
        let mut params = deployment
            .map(|deployment| deployment.defaults.clone())
            .unwrap_or_default();
        params.extend(request.chat_parameters()?);

        Ok(params)
    }

    fn resolve_chat_param_rules(
        &self,
        provider: &ProviderId,
        provider_model: &ModelName,
        adapter: &dyn crate::ChatCompletionAdapter,
    ) -> ResolvedChatParamRules {
        let mut rules = ResolvedChatParamRules::new(adapter.supported_chat_params());
        if let Some(config) = self.inner.provider_chat_params.get(provider) {
            rules.apply_provider_config(config);
            if let Some(model_config) = config.models.get(provider_model) {
                rules.apply_model_config(model_config);
            }
        }
        rules
    }

    fn apply_chat_param_rules(
        &self,
        provider: &ProviderId,
        mut params: ChatParameterMap,
        rules: &ResolvedChatParamRules,
    ) -> SigmaResult<ChatParameterMap> {
        apply_top_level_drops(&mut params, &rules.drop);

        let unsupported = params
            .keys()
            .filter(|param| !rules.supports(param))
            .cloned()
            .collect::<Vec<_>>();

        if !unsupported.is_empty() {
            match rules.policy {
                ParamPolicy::RejectUnsupported => {
                    return Err(SigmaError::UnsupportedParams {
                        provider: provider.clone(),
                        params: unsupported,
                    });
                }
                ParamPolicy::DropUnsupported => {
                    for param in unsupported {
                        params.remove(&param);
                    }
                }
            }
        }

        apply_renames(&mut params, &rules.rename);
        apply_nested_drops(&mut params, &rules.drop);

        Ok(params)
    }

    fn resolve_route(&self, model: &ModelRef) -> SigmaResult<ResolvedRoute> {
        match model {
            ModelRef::ProviderModel { provider, model } => {
                let provider = self.inner.providers.get(provider).cloned().ok_or_else(|| {
                    SigmaError::ProviderConfig {
                        provider: Some(provider.clone()),
                        message: "unknown provider instance".to_string(),
                    }
                })?;

                Ok(ResolvedRoute {
                    provider,
                    deployment: None,
                    public_model: model.clone(),
                    provider_model: model.clone(),
                })
            }
            ModelRef::Deployment(deployment_id) => {
                let deployment = self
                    .inner
                    .deployments_by_id
                    .get(deployment_id)
                    .cloned()
                    .ok_or_else(|| SigmaError::NoDeploymentForModel {
                        model: model.clone(),
                    })?;
                self.route_for_deployment(deployment)
            }
            ModelRef::Model(model_name) => {
                let model_name = if model_name.as_str().is_empty() {
                    self.inner.config.default_model.as_ref().ok_or_else(|| {
                        SigmaError::InvalidArgument(
                            "ChatRequest.model is empty and ClientConfig.default_model is not set"
                                .to_string(),
                        )
                    })?
                } else {
                    model_name
                };

                let deployment_id = self
                    .inner
                    .deployments_by_public_model
                    .get(model_name)
                    .ok_or_else(|| SigmaError::NoDeploymentForModel {
                        model: ModelRef::Model(model_name.clone()),
                    })?;
                let deployment = self
                    .inner
                    .deployments_by_id
                    .get(deployment_id)
                    .cloned()
                    .ok_or_else(|| SigmaError::NoDeploymentForModel {
                        model: ModelRef::Model(model_name.clone()),
                    })?;
                self.route_for_deployment(deployment)
            }
        }
    }

    fn route_for_deployment(
        &self,
        deployment: ModelDeploymentConfig,
    ) -> SigmaResult<ResolvedRoute> {
        let provider = self
            .inner
            .providers
            .get(&deployment.provider)
            .cloned()
            .ok_or_else(|| SigmaError::ProviderConfig {
                provider: Some(deployment.provider.clone()),
                message: format!(
                    "deployment `{}` references an unknown provider instance",
                    deployment.id
                ),
            })?;

        Ok(ResolvedRoute {
            provider,
            public_model: deployment.public_model.clone(),
            provider_model: deployment.provider_model.clone(),
            deployment: Some(deployment),
        })
    }
}

struct ResolvedChatParamRules {
    policy: ParamPolicy,
    supported: HashSet<String>,
    drop: Vec<String>,
    rename: BTreeMap<String, String>,
}

impl ResolvedChatParamRules {
    fn new(supported: Vec<&'static str>) -> Self {
        Self {
            policy: ParamPolicy::RejectUnsupported,
            supported: supported.into_iter().map(str::to_string).collect(),
            drop: Vec::new(),
            rename: BTreeMap::new(),
        }
    }

    fn apply_provider_config(&mut self, config: &ChatParamConfig) {
        if let Some(policy) = config.policy {
            self.policy = policy;
        }
        if let Some(supported) = &config.supported {
            self.supported = supported.iter().cloned().collect();
        }
        self.supported.extend(config.allow.iter().cloned());
        self.drop.extend(config.drop.iter().cloned());
        if let Some(rename) = &config.rename {
            self.rename = rename.clone();
        }
    }

    fn apply_model_config(&mut self, config: &ChatParamModelConfig) {
        if let Some(policy) = config.policy {
            self.policy = policy;
        }
        if let Some(supported) = &config.supported {
            self.supported = supported.iter().cloned().collect();
        }
        self.supported.extend(config.allow.iter().cloned());
        self.drop.extend(config.drop.iter().cloned());
        if let Some(rename) = &config.rename {
            self.rename = rename.clone();
        }
    }

    fn supports(&self, param: &str) -> bool {
        self.supported.contains(param) || self.rename.contains_key(param)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParamPathSegment {
    Key(String),
    Wildcard,
    Index(usize),
}

fn apply_top_level_drops(params: &mut ChatParameterMap, drops: &[String]) {
    for drop in drops {
        if !is_nested_param_path(drop) {
            params.remove(drop);
        }
    }
}

fn apply_nested_drops(params: &mut ChatParameterMap, drops: &[String]) {
    if !drops.iter().any(|drop| is_nested_param_path(drop)) {
        return;
    }

    let mut value = Value::Object(std::mem::take(params));
    for drop in drops {
        if is_nested_param_path(drop) {
            delete_nested_param_path(&mut value, &parse_param_path(drop));
        }
    }

    let Value::Object(map) = value else {
        unreachable!("chat parameters are always represented as a JSON object");
    };
    *params = map;
}

fn apply_renames(params: &mut ChatParameterMap, renames: &BTreeMap<String, String>) {
    for (source, target) in renames {
        if let Some(value) = params.remove(source) {
            params.insert(target.clone(), value);
        }
    }
}

fn is_nested_param_path(path: &str) -> bool {
    path.contains('.') || path.contains('[')
}

fn parse_param_path(path: &str) -> Vec<ParamPathSegment> {
    let mut segments = Vec::new();

    for part in path.split('.') {
        let mut rest = part;
        while let Some(index) = rest.find('[') {
            let key = &rest[..index];
            if !key.is_empty() {
                segments.push(ParamPathSegment::Key(key.to_string()));
            }

            let Some(close) = rest[index..].find(']') else {
                return segments;
            };
            let bracket = &rest[index + 1..index + close];
            if bracket == "*" {
                segments.push(ParamPathSegment::Wildcard);
            } else if let Ok(index) = bracket.parse::<usize>() {
                segments.push(ParamPathSegment::Index(index));
            }
            rest = &rest[index + close + 1..];
        }

        if !rest.is_empty() {
            segments.push(ParamPathSegment::Key(rest.to_string()));
        }
    }

    segments
}

fn delete_nested_param_path(value: &mut Value, segments: &[ParamPathSegment]) {
    let Some((segment, rest)) = segments.split_first() else {
        return;
    };

    match segment {
        ParamPathSegment::Key(key) => {
            let Some(object) = value.as_object_mut() else {
                return;
            };
            if rest.is_empty() {
                object.remove(key);
            } else if let Some(value) = object.get_mut(key) {
                delete_nested_param_path(value, rest);
            }
        }
        ParamPathSegment::Wildcard => {
            let Some(array) = value.as_array_mut() else {
                return;
            };
            if !rest.is_empty() {
                for value in array {
                    delete_nested_param_path(value, rest);
                }
            }
        }
        ParamPathSegment::Index(index) => {
            let Some(array) = value.as_array_mut() else {
                return;
            };
            if !rest.is_empty()
                && let Some(value) = array.get_mut(*index)
            {
                delete_nested_param_path(value, rest);
            }
        }
    }
}

impl ResolvedRoute {
    fn adapter_context(&self) -> ChatAdapterContext<'_> {
        ChatAdapterContext {
            provider: self.provider.id(),
            deployment: self.deployment.as_ref().map(|deployment| &deployment.id),
            public_model: &self.public_model,
            provider_model: &self.provider_model,
            model_info: deployment_model_info(self.deployment.as_ref()),
            provider_state: None,
        }
    }

    fn to_routed_request<'a>(&'a self, request: &'a ChatRequest) -> crate::RoutedChatRequest<'a> {
        crate::RoutedChatRequest {
            provider: self.provider.id(),
            deployment: self.deployment.as_ref().map(|deployment| &deployment.id),
            public_model: &self.public_model,
            provider_model: &self.provider_model,
            model_info: deployment_model_info(self.deployment.as_ref()),
            request,
        }
    }
}

fn http_error(error: reqwest::Error) -> SigmaError {
    SigmaError::Http {
        message: error.to_string(),
    }
}

fn transform_response_or_error(
    adapter: &dyn crate::ChatCompletionAdapter,
    context: &crate::ChatAdapterContext<'_>,
    response: crate::ProviderResponse,
) -> SigmaResult<ChatResponse> {
    if response.status.is_success() {
        adapter.transform_response(context, response)
    } else {
        Err(adapter.transform_error_response(context, response))
    }
}

async fn provider_response(response: reqwest::Response) -> SigmaResult<crate::ProviderResponse> {
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.bytes().await.map_err(http_error)?;

    Ok(crate::ProviderResponse {
        status,
        headers,
        body,
    })
}
