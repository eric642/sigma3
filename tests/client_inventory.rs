#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use futures_util::StreamExt;
use http::{HeaderMap, HeaderValue, Method, StatusCode};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use sigma::types::chat::{
    AssistantDelta, ChatMessage, ChatRequest, ChatResponse, ChatStreamChoice, ChatStreamChunk,
    DeveloperMessage, TextContent, UserMessage,
};
use sigma::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatParameterMap, ChatStream,
    Client, ClientConfig, ModelDeploymentConfig, ModelName, ModelRef, ProviderByteStream,
    ProviderCatalog, ProviderCommonConfig, ProviderConfigMap, ProviderDriver, ProviderEndpoint,
    ProviderId, ProviderInit, ProviderInstanceConfig, ProviderKind, ProviderKindStatic,
    ProviderRequest, ProviderResponse, SecretString, SigmaError, SigmaResult,
    SignedProviderRequest, apply_chat_param_rules, merge_chat_params, provider_registration,
    resolve_chat_param_rules, submit_provider,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

static EVENTS: OnceLock<Mutex<HashMap<String, Vec<String>>>> = OnceLock::new();

fn events() -> &'static Mutex<HashMap<String, Vec<String>>> {
    EVENTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn clear_events(id: &str) {
    events().lock().unwrap().remove(id);
}

fn push_event(id: &ProviderId, event: &str) {
    events()
        .lock()
        .unwrap()
        .entry(id.to_string())
        .or_default()
        .push(event.to_string());
}

fn take_events(id: &str) -> Vec<String> {
    events().lock().unwrap().remove(id).unwrap_or_default()
}

struct FakeProvider {
    id: ProviderId,
    kind: ProviderKind,
    chat: FakeChatAdapter,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
struct FakeProviderConfig {
    #[serde(default = "default_stream_transform")]
    stream_transform: String,
}

impl Default for FakeProviderConfig {
    fn default() -> Self {
        Self {
            stream_transform: default_stream_transform(),
        }
    }
}

fn default_stream_transform() -> String {
    "json".to_string()
}

impl FakeProvider {
    fn from_config(init: ProviderInit<FakeProviderConfig>) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let base_url = init
            .common
            .api_base
            .clone()
            .ok_or_else(|| SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message: "fake provider requires api_base".to_string(),
            })?;
        let config = init.config;

        Ok(Arc::new(Self {
            id: init.id.clone(),
            kind: init.kind,
            chat: FakeChatAdapter {
                id: init.id,
                base_url,
                stream_transform: FakeStreamTransform::from_config(&config.stream_transform),
            },
        }))
    }
}

impl ProviderDriver for FakeProvider {
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

submit_provider! {
    kind: ProviderKindStatic::new("fake-chat"),
    constructor: FakeProvider::from_config,
    config: FakeProviderConfig,
}

struct FakeChatAdapter {
    id: ProviderId,
    base_url: String,
    stream_transform: FakeStreamTransform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FakeStreamTransform {
    Json,
    RawText,
}

impl FakeStreamTransform {
    fn from_config(value: &str) -> Self {
        match value {
            "raw_text" => Self::RawText,
            _ => Self::Json,
        }
    }
}

fn fake_generated_body_key(key: &str) -> bool {
    key == "model" || key == "messages"
}

fn fake_contains_provider_option(provider_options: Option<&ChatParameterMap>, key: &str) -> bool {
    provider_options.is_some_and(|provider_options| provider_options.contains_key(key))
}

fn fake_chat_body_value(
    provider: &ProviderId,
    params: &ChatParameterMap,
    provider_model: &ModelName,
    messages: &[ChatMessage],
    provider_options: Option<&ChatParameterMap>,
) -> SigmaResult<Value> {
    let mut body = serde_json::Map::new();
    for (key, value) in params {
        if !fake_generated_body_key(key) && !fake_contains_provider_option(provider_options, key) {
            body.insert(key.clone(), value.clone());
        }
    }
    if !fake_contains_provider_option(provider_options, "model") {
        body.insert(
            "model".to_string(),
            Value::String(provider_model.to_string()),
        );
    }
    if !fake_contains_provider_option(provider_options, "messages") {
        body.insert(
            "messages".to_string(),
            fake_messages_to_value(provider, messages)?,
        );
    }
    if let Some(provider_options) = provider_options {
        body.extend(provider_options.clone());
    }

    Ok(Value::Object(body))
}

fn fake_messages_to_value(provider: &ProviderId, messages: &[ChatMessage]) -> SigmaResult<Value> {
    let translated = messages
        .iter()
        .map(|message| match message {
            ChatMessage::Developer(message) => {
                let content = match &message.content {
                    TextContent::Text(text) => Value::String(text.clone()),
                    TextContent::Parts(parts) => serde_json::to_value(parts)?,
                };

                Ok(json!({
                    "role": "system",
                    "content": content,
                }))
            }
            other => serde_json::to_value(other),
        })
        .collect::<Result<Vec<_>, serde_json::Error>>()
        .map_err(|err| SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: err.to_string(),
        })?;

    Ok(Value::Array(translated))
}

const FAKE_SUPPORTED_CHAT_PARAMS: &[&str] = &["temperature", "stream", "max_completion_tokens"];

impl ChatCompletionAdapter for FakeChatAdapter {
    fn endpoint(&self, _request: &ChatAdapterRequest<'_>) -> SigmaResult<ProviderEndpoint> {
        push_event(&self.id, "endpoint");
        Ok(ProviderEndpoint {
            method: Method::POST,
            url: format!("{}/chat", self.base_url.trim_end_matches('/')),
        })
    }

    fn transform_request(
        &self,
        request: ChatAdapterRequest<'_>,
        endpoint: ProviderEndpoint,
    ) -> SigmaResult<ProviderRequest> {
        push_event(&self.id, "transform_request");
        let mut params = merge_chat_params(
            request.deployment_defaults,
            request.request,
            request.streaming,
        )?;
        let rules = resolve_chat_param_rules(
            FAKE_SUPPORTED_CHAT_PARAMS,
            None,
            request.context.provider_model,
        );
        apply_chat_param_rules(&self.id, &mut params, &rules)?;

        let provider_options = request.request.provider_options.get(&self.id);
        let body = fake_chat_body_value(
            &self.id,
            &params,
            request.context.provider_model,
            &request.request.messages,
            provider_options,
        )?;

        Ok(ProviderRequest {
            method: endpoint.method,
            url: endpoint.url,
            headers: HeaderMap::new(),
            body,
            provider_state: None,
        })
    }

    fn sign_request(&self, mut request: ProviderRequest) -> SigmaResult<SignedProviderRequest> {
        push_event(&self.id, "sign_request");
        request.headers.insert(
            "x-provider-id",
            HeaderValue::from_str(self.id.as_str()).unwrap(),
        );

        Ok(request.into())
    }

    fn transform_response(
        &self,
        _context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaResult<ChatResponse> {
        push_event(&self.id, "transform_response");
        serde_json::from_slice(&response.body).map_err(|err| SigmaError::ProviderResponse {
            provider: self.id.clone(),
            message: err.to_string(),
        })
    }

    fn transform_error_response(
        &self,
        _context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaError {
        push_event(&self.id, "transform_error_response");

        let error = serde_json::from_slice::<Value>(&response.body)
            .ok()
            .and_then(|body| body.get("error").cloned())
            .unwrap_or(Value::Null);
        let code = error
            .get("code")
            .and_then(Value::as_str)
            .map(str::to_string);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("provider error")
            .to_string();
        let details = error.get("details").cloned();

        SigmaError::ProviderBusiness {
            provider: self.id.clone(),
            status: response.status,
            code,
            message,
            details,
        }
    }

    fn transform_stream(
        &self,
        _context: &ChatAdapterContext<'_>,
        stream: ProviderByteStream,
    ) -> SigmaResult<ChatStream> {
        push_event(&self.id, "transform_stream");
        let provider = self.id.clone();

        match self.stream_transform {
            FakeStreamTransform::Json => Ok(Box::pin(stream.map(move |chunk| {
                let chunk = chunk?;
                serde_json::from_slice(&chunk).map_err(|err| SigmaError::ProviderResponse {
                    provider: provider.clone(),
                    message: err.to_string(),
                })
            }))),
            FakeStreamTransform::RawText => Ok(Box::pin(stream.map(move |chunk| {
                let chunk = chunk?;
                let text = String::from_utf8(chunk.to_vec()).map_err(|err| {
                    SigmaError::ProviderResponse {
                        provider: provider.clone(),
                        message: err.to_string(),
                    }
                })?;

                Ok(stream_chunk(
                    "custom",
                    "custom-model",
                    &format!("custom:{text}"),
                ))
            }))),
        }
    }
}

fn provider_from_request(request: &WiremockRequest) -> ProviderId {
    let provider = request
        .headers
        .get("x-provider-id")
        .and_then(|value| value.to_str().ok())
        .unwrap();
    ProviderId::from(provider)
}

fn request_model(request: &WiremockRequest) -> String {
    let body: Value = request.body_json().unwrap();
    body.get("model")
        .and_then(Value::as_str)
        .unwrap()
        .to_string()
}

async fn mount_chat_response(server: &MockServer, event: &'static str, content: &'static str) {
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(move |request: &WiremockRequest| {
            let provider = provider_from_request(request);
            push_event(&provider, event);
            let model = request_model(request);
            ResponseTemplate::new(200).set_body_json(response_body(&model, content))
        })
        .mount(server)
        .await;
}

async fn mount_bytes_response(server: &MockServer, event: &'static str, body: Vec<u8>) {
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(move |request: &WiremockRequest| {
            let provider = provider_from_request(request);
            push_event(&provider, event);
            ResponseTemplate::new(200).set_body_bytes(body.clone())
        })
        .mount(server)
        .await;
}

async fn mount_error_response(server: &MockServer, event: &'static str) {
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(move |request: &WiremockRequest| {
            let provider = provider_from_request(request);
            push_event(&provider, event);
            ResponseTemplate::new(StatusCode::TOO_MANY_REQUESTS.as_u16())
                .set_body_json(error_body())
        })
        .mount(server)
        .await;
}

async fn last_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap();
    requests.last().unwrap().body_json().unwrap()
}

fn response_body(model: &str, content: &str) -> Value {
    json!({
        "id": "chatcmpl_fake",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "created": 1,
        "model": model,
        "object": "chat.completion",
        "usage": null,
    })
}

fn error_body() -> Value {
    json!({
        "error": {
            "code": "rate_limit",
            "message": "too many requests",
            "details": {
                "retry_after": 60,
            },
        },
    })
}

fn stream_chunk(id: &str, model: &str, content: &str) -> ChatStreamChunk {
    ChatStreamChunk {
        id: id.to_string(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: AssistantDelta {
                content: Some(content.to_string()),
                reasoning: None,
                tool_calls: None,
                role: None,
                refusal: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
        created: 1,
        model: model.to_string(),
        service_tier: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
    }
}

fn stream_chunk_bytes(model: &str, content: &str) -> Vec<u8> {
    serde_json::to_vec(&stream_chunk("chunk_fake", model, content)).unwrap()
}

fn provider_config_map(value: Value) -> ProviderConfigMap {
    match value {
        Value::Object(map) => map,
        Value::Null => ProviderConfigMap::new(),
        other => panic!("provider config must be an object or null, got {other:?}"),
    }
}

fn config(provider_id: &str, api_base: &str, provider_config: Value) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: provider_id.into(),
            kind: ProviderKind::from("fake-chat"),
            common: ProviderCommonConfig {
                api_base: Some(api_base.to_string()),
                api_key: None,
                headers: HashMap::new(),
            },
            config: provider_config_map(provider_config),
        }],
        deployments: vec![ModelDeploymentConfig {
            id: "dep-chat".into(),
            public_model: "gpt-public".into(),
            provider: provider_id.into(),
            provider_model: "provider-gpt".into(),
            defaults: serde_json::Map::new(),
            model_info: Value::Null,
        }],
        default_model: None,
    }
}

fn assert_provider_config_from_serde(config: &ClientConfig) {
    let provider = config.providers.first().unwrap();

    assert_eq!(provider.id, ProviderId::from("configured"));
    assert_eq!(provider.kind, ProviderKind::from("fake-chat"));
    assert_eq!(
        provider.common.api_base.as_deref(),
        Some("http://localhost:8080/v1")
    );
    assert_eq!(
        provider
            .common
            .api_key
            .as_ref()
            .map(SecretString::expose_secret),
        Some("sk-test")
    );
    assert_eq!(
        provider.common.headers.get("X-Test").map(String::as_str),
        Some("yes")
    );
    assert_eq!(provider.config["stream_transform"], "raw_text");
}

#[test]
fn client_config_deserializes_nested_provider_config_from_serde_formats() {
    let json_config = r#"{
        "providers": [{
            "id": "configured",
            "kind": "fake-chat",
            "api_base": "http://localhost:8080/v1",
            "api_key": "sk-test",
            "headers": { "X-Test": "yes" },
            "config": { "stream_transform": "raw_text" }
        }]
    }"#;
    let toml_config = r#"
        [[providers]]
        id = "configured"
        kind = "fake-chat"
        api_base = "http://localhost:8080/v1"
        api_key = "sk-test"

        [providers.headers]
        X-Test = "yes"

        [providers.config]
        stream_transform = "raw_text"
    "#;
    let yaml_config = r#"
providers:
  - id: configured
    kind: fake-chat
    api_base: http://localhost:8080/v1
    api_key: sk-test
    headers:
      X-Test: "yes"
    config:
      stream_transform: raw_text
"#;

    assert_provider_config_from_serde(&serde_json::from_str(json_config).unwrap());
    assert_provider_config_from_serde(&toml::from_str(toml_config).unwrap());
    assert_provider_config_from_serde(&serde_yaml::from_str(yaml_config).unwrap());
}

#[test]
fn provider_init_deserialize_config_rejects_unknown_provider_config_fields() {
    let init = ProviderInit::from(ProviderInstanceConfig {
        id: ProviderId::from("configured"),
        kind: ProviderKind::from("fake-chat"),
        common: ProviderCommonConfig {
            api_base: Some("http://localhost:8080/v1".to_string()),
            api_key: None,
            headers: HashMap::new(),
        },
        config: provider_config_map(json!({
            "unknown": true
        })),
    });

    let err = match init.into_typed_config::<FakeProviderConfig>() {
        Ok(_) => panic!("expected provider config error"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        SigmaError::ProviderConfig {
            provider: Some(provider),
            message
        } if provider == "configured" && message.contains("unknown field")
    ));
}

fn request(model: ModelRef) -> sigma::types::chat::ChatRequest {
    ChatRequest::new(model, vec![UserMessage::from("hello").into()])
}

fn client(provider_id: &str, server: &MockServer, provider_config: Value) -> Client {
    clear_events(provider_id);
    Client::build(config(provider_id, &server.uri(), provider_config)).unwrap()
}

fn duplicate_constructor<TConfig>(
    _init: ProviderInit<TConfig>,
) -> SigmaResult<Arc<dyn ProviderDriver>> {
    unreachable!("duplicate registration detection should not call constructors")
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
struct DuplicateConfig {}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
#[allow(dead_code)]
struct AlphaConfig {
    #[serde(default)]
    feature: bool,
    #[serde(default)]
    mode: AlphaMode,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AlphaMode {
    #[default]
    Standard,
    Strict,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
struct ZetaConfig {}

#[test]
fn catalog_from_inventory_collects_fake_provider_registration() {
    let catalog = ProviderCatalog::from_inventory().unwrap();

    assert!(catalog.contains_kind(&ProviderKind::from("fake-chat")));
}

#[test]
fn catalog_collects_sorted_provider_instance_config_schemas() {
    let catalog = ProviderCatalog::from_registrations([
        provider_registration! {
            kind: ProviderKindStatic::new("zeta"),
            constructor: duplicate_constructor,
            config: ZetaConfig,
        },
        provider_registration! {
            kind: ProviderKindStatic::new("alpha"),
            constructor: duplicate_constructor,
            config: AlphaConfig,
        },
    ])
    .unwrap();

    let schemas = catalog.provider_instance_config_schemas();

    assert_eq!(schemas[0].kind, ProviderKind::from("alpha"));
    assert_eq!(schemas[1].kind, ProviderKind::from("zeta"));
    assert_eq!(schemas[0].schema["properties"]["kind"]["const"], "alpha");
    assert_eq!(
        schemas[0].schema["properties"]["api_base"]["type"],
        "string"
    );
    assert_eq!(
        schemas[0].schema["properties"]["headers"]["additionalProperties"]["type"],
        "string"
    );
    assert_eq!(
        schemas[0].schema["properties"]["config"]["properties"]["feature"]["type"],
        "boolean"
    );
    assert_eq!(
        schemas[0].schema["properties"]["config"]["properties"]["feature"]["default"],
        false
    );
    assert_eq!(
        schemas[0].schema["properties"]["config"]["properties"]["mode"]["enum"],
        json!(["standard", "strict"])
    );
}

#[test]
fn catalog_from_registrations_rejects_duplicate_kind() {
    let registrations = [
        provider_registration! {
            kind: ProviderKindStatic::new("duplicate"),
            constructor: duplicate_constructor,
            config: DuplicateConfig,
        },
        provider_registration! {
            kind: ProviderKindStatic::new("duplicate"),
            constructor: duplicate_constructor,
            config: DuplicateConfig,
        },
    ];

    let err = ProviderCatalog::from_registrations(registrations).unwrap_err();

    assert!(matches!(
        err,
        SigmaError::DuplicateProviderRegistration { kind } if kind == "duplicate"
    ));
}

#[test]
fn provider_request_body_is_structured_json() {
    let request = ProviderRequest {
        method: Method::POST,
        url: "http://localhost/chat".to_string(),
        headers: HeaderMap::new(),
        body: json!({"provider_native": true}),
        provider_state: None,
    };

    let signed = SignedProviderRequest::from(request);

    assert_eq!(signed.body["provider_native"], true);
}

#[tokio::test]
async fn create_routes_public_model_to_provider_model() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client("p-public", &server, Value::Null);

    let response = client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    assert_eq!(response.model, "provider-gpt");
}

#[tokio::test]
async fn create_routes_deployment_id_to_provider_model() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client("p-deployment", &server, Value::Null);

    let response = client
        .chat()
        .create(&request(ModelRef::deployment("dep-chat")))
        .await
        .unwrap();

    assert_eq!(response.model, "provider-gpt");
}

#[tokio::test]
async fn create_returns_invalid_argument_when_model_empty_and_default_unset() {
    let server = MockServer::start().await;
    let provider_id = "p-empty-model";
    let client = client(provider_id, &server, Value::Null);

    let err = client
        .chat()
        .create(&request(ModelRef::default()))
        .await
        .expect_err("empty model with no default_model should fail with InvalidArgument");

    assert!(matches!(
        err,
        SigmaError::InvalidArgument(ref message)
            if message.contains("default_model")
    ));
}

#[tokio::test]
async fn create_routes_provider_model_directly() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let provider_id = "p-direct";
    let mut config = config(provider_id, &server.uri(), Value::Null);
    config.deployments.clear();

    let client = Client::builder()
        .with_http_client(reqwest::Client::new())
        .build(config)
        .unwrap();

    let response = client
        .chat()
        .create(&request(ModelRef::provider_model(
            provider_id,
            "direct-model",
        )))
        .await
        .unwrap();

    assert_eq!(response.model, "direct-model");
}

#[tokio::test]
async fn create_accepts_borrowed_request_without_consuming_it() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client("p-borrowed-create", &server, Value::Null);
    let request = request(ModelRef::model("gpt-public"));

    client.chat().create(&request).await.unwrap();

    assert_eq!(request.model, ModelRef::model("gpt-public"));
}

#[tokio::test]
async fn create_stream_accepts_borrowed_request_without_consuming_it() {
    let server = MockServer::start().await;
    mount_bytes_response(
        &server,
        "http.stream",
        stream_chunk_bytes("provider-gpt", "chunk"),
    )
    .await;
    let client = client("p-borrowed-stream", &server, Value::Null);
    let request = request(ModelRef::model("gpt-public"));

    let mut stream = client.chat().create_stream(&request).await.unwrap();
    let _ = stream.next().await.unwrap().unwrap();

    assert_eq!(request.model, ModelRef::model("gpt-public"));
}

#[tokio::test]
async fn create_runs_adapter_lifecycle_in_order() {
    let server = MockServer::start().await;
    let provider_id = "p-lifecycle";
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(provider_id, &server, Value::Null);

    client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    assert_eq!(
        take_events(provider_id),
        vec![
            "endpoint",
            "transform_request",
            "sign_request",
            "http.execute",
            "transform_response",
        ]
    );
}

#[tokio::test]
async fn create_lets_adapter_transform_non_success_status_into_business_error() {
    let server = MockServer::start().await;
    let provider_id = "p-create-error";
    mount_error_response(&server, "http.execute").await;
    let client = client(provider_id, &server, Value::Null);

    let err = client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        SigmaError::ProviderBusiness {
            provider,
            status,
            code: Some(code),
            message,
            details: Some(details),
        } if provider == provider_id
            && status == StatusCode::TOO_MANY_REQUESTS
            && code == "rate_limit"
            && message == "too many requests"
            && details == json!({"retry_after": 60})
    ));
    assert_eq!(
        take_events(provider_id),
        vec![
            "endpoint",
            "transform_request",
            "sign_request",
            "http.execute",
            "transform_error_response",
        ]
    );
}

#[tokio::test]
async fn create_rejects_unsupported_params_when_policy_rejects() {
    let server = MockServer::start().await;
    let client = client("p-reject", &server, Value::Null);

    let mut request = request(ModelRef::model("gpt-public"));
    request.params.count = Some(2);
    let err = client.chat().create(&request).await.unwrap_err();

    assert!(matches!(
        err,
        SigmaError::UnsupportedParams { params, .. } if params == vec!["count"]
    ));
}

#[tokio::test]
async fn create_applies_selected_provider_options_after_adapter_mapping() {
    let server = MockServer::start().await;
    let provider_id = "zhipu";
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(provider_id, &server, Value::Null);
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.temperature = Some(0.2);
    let mut overrides = ChatParameterMap::new();
    overrides.insert("model".to_string(), json!("override-model"));
    overrides.insert("temperature".to_string(), json!(0.9));
    overrides.insert("provider_native".to_string(), json!(true));
    request
        .provider_options
        .insert(ProviderId::from(provider_id), overrides);

    let response = client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(response.model, "override-model");
    assert_eq!(body["model"], "override-model");
    assert_eq!(body["temperature"], json!(0.9));
    assert_eq!(body["provider_native"], json!(true));
}

#[tokio::test]
async fn create_ignores_provider_options_for_non_selected_provider() {
    let server = MockServer::start().await;
    let provider_id = "selected-provider";
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(provider_id, &server, Value::Null);
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.temperature = Some(0.2);
    let mut overrides = ChatParameterMap::new();
    overrides.insert("temperature".to_string(), json!(0.9));
    overrides.insert("provider_native".to_string(), json!(true));
    request
        .provider_options
        .insert(ProviderId::from("other-provider"), overrides);

    client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert!((body["temperature"].as_f64().unwrap() - 0.2).abs() < 0.000001);
    assert!(body.get("provider_native").is_none());
}

#[tokio::test]
async fn create_keeps_provider_body_structured_for_provider_options() {
    let server = MockServer::start().await;
    let provider_id = "p-structured-body";
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(provider_id, &server, Value::Null);
    let mut request = request(ModelRef::model("gpt-public"));
    let mut overrides = ChatParameterMap::new();
    overrides.insert("provider_native".to_string(), json!(true));
    request
        .provider_options
        .insert(ProviderId::from(provider_id), overrides);

    let response = client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(response.model, "provider-gpt");
    assert_eq!(body["provider_native"], true);
}

#[tokio::test]
async fn create_lets_adapter_transform_developer_messages_in_request_body() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client("p-developer", &server, Value::Null);

    let request = ChatRequest::new(
        ModelRef::model("gpt-public"),
        vec![
            DeveloperMessage {
                content: TextContent::Text("developer instruction".to_string()),
                name: None,
            }
            .into(),
        ],
    );

    client.chat().create(&request).await.unwrap();

    assert_eq!(last_body(&server).await["messages"][0]["role"], "system");
}

#[tokio::test]
async fn create_stream_provider_options_can_override_injected_stream_param() {
    let server = MockServer::start().await;
    let provider_id = "p-stream-override";
    mount_bytes_response(
        &server,
        "http.stream",
        stream_chunk_bytes("provider-gpt", "chunk"),
    )
    .await;
    let client = client(provider_id, &server, Value::Null);
    let mut request = request(ModelRef::model("gpt-public"));
    let mut overrides = ChatParameterMap::new();
    overrides.insert("stream".to_string(), json!(false));
    request
        .provider_options
        .insert(ProviderId::from(provider_id), overrides);

    let mut stream = client.chat().create_stream(&request).await.unwrap();
    let _ = stream.next().await.unwrap().unwrap();

    assert_eq!(last_body(&server).await["stream"], json!(false));
}

#[tokio::test]
async fn create_stream_injects_stream_param_for_native_streams() {
    let server = MockServer::start().await;
    let provider_id = "p-stream-native";
    mount_bytes_response(
        &server,
        "http.stream",
        stream_chunk_bytes("provider-gpt", "chunk"),
    )
    .await;
    let client = client(provider_id, &server, Value::Null);

    let mut stream = client
        .chat()
        .create_stream(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();
    let _ = stream.next().await.unwrap().unwrap();

    assert_eq!(last_body(&server).await["stream"], true);
    assert_eq!(
        take_events(provider_id),
        vec![
            "endpoint",
            "transform_request",
            "sign_request",
            "http.stream",
            "transform_stream",
        ]
    );
}

#[tokio::test]
async fn create_stream_uses_adapter_stream_transform_for_provider_bytes() {
    let server = MockServer::start().await;
    mount_bytes_response(&server, "http.stream", b"raw".to_vec()).await;
    let client = client(
        "p-stream-transform",
        &server,
        json!({"stream_transform": "raw_text"}),
    );

    let mut stream = client
        .chat()
        .create_stream(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();
    let chunk = stream.next().await.unwrap().unwrap();

    assert_eq!(
        chunk.choices[0].delta.content.as_deref(),
        Some("custom:raw")
    );
}

#[tokio::test]
async fn create_stream_native_lets_adapter_transform_non_success_status_into_business_error() {
    let server = MockServer::start().await;
    let provider_id = "p-stream-native-error";
    mount_error_response(&server, "http.stream").await;
    let client = client(provider_id, &server, Value::Null);

    let err = match client
        .chat()
        .create_stream(&request(ModelRef::model("gpt-public")))
        .await
    {
        Ok(_) => panic!("expected provider business error"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        SigmaError::ProviderBusiness {
            provider,
            status,
            code: Some(code),
            message,
            details: Some(details),
        } if provider == provider_id
            && status == StatusCode::TOO_MANY_REQUESTS
            && code == "rate_limit"
            && message == "too many requests"
            && details == json!({"retry_after": 60})
    ));
    assert_eq!(
        take_events(provider_id),
        vec![
            "endpoint",
            "transform_request",
            "sign_request",
            "http.stream",
            "transform_error_response",
        ]
    );
}
