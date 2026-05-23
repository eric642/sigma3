use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bytes::Bytes;
use futures_util::StreamExt;
use http::{HeaderMap, HeaderValue, Method, StatusCode};
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};
use serde_json::{Value, json};
use sigma::types::chat::{
    ChatChoiceStream, ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestDeveloperMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage, ChatCompletionStreamResponseDelta,
    CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
};
use sigma::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatParameterMap, ChatStream,
    Client, ClientConfig, ModelDeploymentConfig, ModelName, ModelRef, ParamPolicy,
    ProviderByteStream, ProviderCatalog, ProviderDriver, ProviderEndpoint, ProviderId,
    ProviderInit, ProviderInstanceConfig, ProviderKind, ProviderKindStatic, ProviderRegistration,
    ProviderRequest, ProviderResponse, SigmaError, SigmaResult, SignedProviderRequest,
    StreamBehavior, submit_provider,
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

impl FakeProvider {
    fn from_config(init: ProviderInit) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let base_url = init
            .api_base
            .clone()
            .ok_or_else(|| SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message: "fake provider requires api_base".to_string(),
            })?;
        let stream_behavior = init
            .options
            .get("stream_behavior")
            .and_then(Value::as_str)
            .unwrap_or("native");

        let stream_behavior = match stream_behavior {
            "fake" => StreamBehavior::fake_from_response(),
            _ => StreamBehavior::native(true),
        };
        let stream_transform = init
            .options
            .get("stream_transform")
            .and_then(Value::as_str)
            .unwrap_or("json");
        let request_body = init
            .options
            .get("request_body")
            .and_then(Value::as_str)
            .unwrap_or("json");

        Ok(Arc::new(Self {
            id: init.id.clone(),
            kind: init.kind,
            chat: FakeChatAdapter {
                id: init.id,
                base_url,
                stream_behavior,
                stream_transform: FakeStreamTransform::from_config(stream_transform),
                request_body: FakeRequestBody::from_config(request_body),
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
}

struct FakeChatAdapter {
    id: ProviderId,
    base_url: String,
    stream_behavior: StreamBehavior,
    stream_transform: FakeStreamTransform,
    request_body: FakeRequestBody,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FakeRequestBody {
    Json,
    RawText,
}

impl FakeRequestBody {
    fn from_config(value: &str) -> Self {
        match value {
            "raw_text" => Self::RawText,
            _ => Self::Json,
        }
    }
}

struct FakeChatBody<'a> {
    params: &'a ChatParameterMap,
    provider_model: &'a ModelName,
    messages: &'a Value,
    body_overrides: Option<&'a ChatParameterMap>,
}

impl Serialize for FakeChatBody<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut len = self
            .params
            .keys()
            .filter(|key| {
                !fake_generated_body_key(key.as_str())
                    && !fake_contains_body_override(self.body_overrides, key.as_str())
            })
            .count();

        if !fake_contains_body_override(self.body_overrides, "model") {
            len += 1;
        }
        if !fake_contains_body_override(self.body_overrides, "messages") {
            len += 1;
        }
        if let Some(body_overrides) = self.body_overrides {
            len += body_overrides.len();
        }

        let mut map = serializer.serialize_map(Some(len))?;
        for (key, value) in self.params {
            if !fake_generated_body_key(key.as_str())
                && !fake_contains_body_override(self.body_overrides, key.as_str())
            {
                map.serialize_entry(key, value)?;
            }
        }
        if !fake_contains_body_override(self.body_overrides, "model") {
            map.serialize_entry("model", self.provider_model)?;
        }
        if !fake_contains_body_override(self.body_overrides, "messages") {
            map.serialize_entry("messages", self.messages)?;
        }
        if let Some(body_overrides) = self.body_overrides {
            for (key, value) in body_overrides {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

fn fake_generated_body_key(key: &str) -> bool {
    key == "model" || key == "messages"
}

fn fake_contains_body_override(body_overrides: Option<&ChatParameterMap>, key: &str) -> bool {
    body_overrides.is_some_and(|body_overrides| body_overrides.contains_key(key))
}

impl ChatCompletionAdapter for FakeChatAdapter {
    fn supported_openai_params(&self) -> Vec<&'static str> {
        push_event(&self.id, "supported_openai_params");
        vec!["temperature", "stream", "max_completion_tokens"]
    }

    fn translate_messages(&self, messages: &[ChatCompletionRequestMessage]) -> SigmaResult<Value> {
        push_event(&self.id, "translate_messages");

        let translated = messages
            .iter()
            .map(|message| match message {
                ChatCompletionRequestMessage::Developer(message) => {
                    let content = match &message.content {
                        ChatCompletionRequestDeveloperMessageContent::Text(text) => {
                            Value::String(text.clone())
                        }
                        ChatCompletionRequestDeveloperMessageContent::Array(parts) => {
                            serde_json::to_value(parts)?
                        }
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
                provider: self.id.clone(),
                message: err.to_string(),
            })?;

        Ok(Value::Array(translated))
    }

    fn map_openai_params(&self, params: ChatParameterMap) -> SigmaResult<ChatParameterMap> {
        push_event(&self.id, "map_openai_params");
        Ok(params)
    }

    fn validate_environment(&self) -> SigmaResult<()> {
        push_event(&self.id, "validate_environment");
        Ok(())
    }

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
        if self.request_body == FakeRequestBody::RawText {
            let suffix = request
                .body_overrides
                .and_then(|body_overrides| body_overrides.get("provider_native"))
                .and_then(Value::as_bool)
                .map(|enabled| format!("provider_native={enabled}"))
                .unwrap_or_else(|| "no-overrides".to_string());

            return Ok(ProviderRequest {
                method: endpoint.method,
                url: endpoint.url,
                headers: HeaderMap::new(),
                body: Bytes::from(format!("raw-body:{suffix}")),
            });
        }

        let body = serde_json::to_vec(&FakeChatBody {
            params: &request.params,
            provider_model: request.context.provider_model,
            messages: &request.messages,
            body_overrides: request.body_overrides,
        })
        .map_err(|err| SigmaError::ProviderTransform {
            provider: self.id.clone(),
            message: err.to_string(),
        })?;

        Ok(ProviderRequest {
            method: endpoint.method,
            url: endpoint.url,
            headers: HeaderMap::new(),
            body: Bytes::from(body),
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
    ) -> SigmaResult<CreateChatCompletionResponse> {
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

    fn stream_behavior(&self) -> StreamBehavior {
        self.stream_behavior
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

async fn mount_static_chat_response(
    server: &MockServer,
    event: &'static str,
    model: &'static str,
    content: &'static str,
) {
    Mock::given(method("POST"))
        .and(path("/chat"))
        .respond_with(move |request: &WiremockRequest| {
            let provider = provider_from_request(request);
            push_event(&provider, event);
            ResponseTemplate::new(200).set_body_json(response_body(model, content))
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

async fn last_raw_body(server: &MockServer) -> Vec<u8> {
    let requests = server.received_requests().await.unwrap();
    requests.last().unwrap().body.clone()
}

async fn request_count(server: &MockServer) -> usize {
    server.received_requests().await.unwrap().len()
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

fn stream_chunk(id: &str, model: &str, content: &str) -> CreateChatCompletionStreamResponse {
    CreateChatCompletionStreamResponse {
        id: id.to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: Some(content.to_string()),
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

fn config(
    provider_id: &str,
    api_base: &str,
    options: Value,
    param_policy: ParamPolicy,
) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: provider_id.into(),
            kind: ProviderKind::from("fake-chat"),
            api_base: Some(api_base.to_string()),
            api_key: None,
            headers: HashMap::new(),
            options,
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
        param_policy,
    }
}

fn request(model: ModelRef) -> sigma::types::chat::CreateChatCompletionRequest {
    CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage::from("hello"),
        )])
        .model(model)
        .build()
        .unwrap()
}

fn client(
    provider_id: &str,
    server: &MockServer,
    options: Value,
    param_policy: ParamPolicy,
) -> Client {
    clear_events(provider_id);
    Client::build(config(provider_id, &server.uri(), options, param_policy)).unwrap()
}

fn duplicate_constructor(_init: ProviderInit) -> SigmaResult<Arc<dyn ProviderDriver>> {
    unreachable!("duplicate registration detection should not call constructors")
}

#[test]
fn catalog_from_inventory_collects_fake_provider_registration() {
    let catalog = ProviderCatalog::from_inventory().unwrap();

    assert!(catalog.contains_kind(&ProviderKind::from("fake-chat")));
}

#[test]
fn catalog_from_registrations_rejects_duplicate_kind() {
    let registrations = [
        ProviderRegistration {
            kind: ProviderKindStatic::new("duplicate"),
            constructor: duplicate_constructor,
        },
        ProviderRegistration {
            kind: ProviderKindStatic::new("duplicate"),
            constructor: duplicate_constructor,
        },
    ];

    let err = ProviderCatalog::from_registrations(registrations).unwrap_err();

    assert!(matches!(
        err,
        SigmaError::DuplicateProviderRegistration { kind } if kind == "duplicate"
    ));
}

#[tokio::test]
async fn create_routes_public_model_to_provider_model() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(
        "p-public",
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );

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
    let client = client(
        "p-deployment",
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );

    let response = client
        .chat()
        .create(&request(ModelRef::deployment("dep-chat")))
        .await
        .unwrap();

    assert_eq!(response.model, "provider-gpt");
}

#[tokio::test]
async fn create_routes_provider_model_directly() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let provider_id = "p-direct";
    let mut config = config(
        provider_id,
        &server.uri(),
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );
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
    let client = client(
        "p-borrowed-create",
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );
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
    let client = client(
        "p-borrowed-stream",
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );
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
    let client = client(
        provider_id,
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );

    client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    assert_eq!(
        take_events(provider_id),
        vec![
            "supported_openai_params",
            "translate_messages",
            "map_openai_params",
            "validate_environment",
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
    let client = client(
        provider_id,
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );

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
            "supported_openai_params",
            "translate_messages",
            "map_openai_params",
            "validate_environment",
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
    let client = client(
        "p-reject",
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );

    let mut request = request(ModelRef::model("gpt-public"));
    request.params.n = Some(2);
    let err = client.chat().create(&request).await.unwrap_err();

    assert!(matches!(
        err,
        SigmaError::UnsupportedParams { params, .. } if params == vec!["n"]
    ));
}

#[tokio::test]
async fn create_drops_unsupported_params_when_policy_drops() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client("p-drop", &server, Value::Null, ParamPolicy::DropUnsupported);

    let mut request = request(ModelRef::model("gpt-public"));
    request.params.n = Some(2);
    client.chat().create(&request).await.unwrap();

    assert!(last_body(&server).await.get("n").is_none());
}

#[tokio::test]
async fn create_applies_selected_provider_metadata_after_adapter_mapping() {
    let server = MockServer::start().await;
    let provider_id = "zhipu";
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(
        provider_id,
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.temperature = Some(0.2);
    let mut overrides = ChatParameterMap::new();
    overrides.insert("model".to_string(), json!("override-model"));
    overrides.insert("temperature".to_string(), json!(0.9));
    overrides.insert("provider_native".to_string(), json!(true));
    request
        .metadata
        .insert(ProviderId::from(provider_id), overrides);

    let response = client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(response.model, "override-model");
    assert_eq!(body["model"], "override-model");
    assert_eq!(body["temperature"], json!(0.9));
    assert_eq!(body["provider_native"], json!(true));
}

#[tokio::test]
async fn create_ignores_metadata_for_non_selected_provider() {
    let server = MockServer::start().await;
    let provider_id = "selected-provider";
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(
        provider_id,
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.temperature = Some(0.2);
    let mut overrides = ChatParameterMap::new();
    overrides.insert("temperature".to_string(), json!(0.9));
    overrides.insert("provider_native".to_string(), json!(true));
    request
        .metadata
        .insert(ProviderId::from("other-provider"), overrides);

    client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert!((body["temperature"].as_f64().unwrap() - 0.2).abs() < 0.000001);
    assert!(body.get("provider_native").is_none());
}

#[tokio::test]
async fn create_does_not_reparse_serialized_provider_body_for_metadata() {
    let server = MockServer::start().await;
    let provider_id = "p-raw-body";
    mount_static_chat_response(&server, "http.execute", "raw-model", "ok").await;
    let client = client(
        provider_id,
        &server,
        json!({"request_body": "raw_text"}),
        ParamPolicy::RejectUnsupported,
    );
    let mut request = request(ModelRef::model("gpt-public"));
    let mut overrides = ChatParameterMap::new();
    overrides.insert("provider_native".to_string(), json!(true));
    request
        .metadata
        .insert(ProviderId::from(provider_id), overrides);

    let response = client.chat().create(&request).await.unwrap();

    assert_eq!(response.model, "raw-model");
    assert_eq!(
        last_raw_body(&server).await,
        b"raw-body:provider_native=true"
    );
}

#[tokio::test]
async fn create_lets_adapter_translate_developer_messages() {
    let server = MockServer::start().await;
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(
        "p-developer",
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );

    let request = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::Developer(
            ChatCompletionRequestDeveloperMessage {
                content: ChatCompletionRequestDeveloperMessageContent::Text(
                    "developer instruction".to_string(),
                ),
                name: None,
            },
        )])
        .model(ModelRef::model("gpt-public"))
        .build()
        .unwrap();

    client.chat().create(&request).await.unwrap();

    assert_eq!(last_body(&server).await["messages"][0]["role"], "system");
}

#[tokio::test]
async fn create_stream_metadata_can_override_injected_stream_param() {
    let server = MockServer::start().await;
    let provider_id = "p-stream-override";
    mount_bytes_response(
        &server,
        "http.stream",
        stream_chunk_bytes("provider-gpt", "chunk"),
    )
    .await;
    let client = client(
        provider_id,
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );
    let mut request = request(ModelRef::model("gpt-public"));
    let mut overrides = ChatParameterMap::new();
    overrides.insert("stream".to_string(), json!(false));
    request
        .metadata
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
    let client = client(
        provider_id,
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );

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
            "supported_openai_params",
            "translate_messages",
            "map_openai_params",
            "validate_environment",
            "endpoint",
            "transform_request",
            "sign_request",
            "http.stream",
            "transform_stream",
        ]
    );
}

#[tokio::test]
async fn create_stream_can_fake_stream_from_non_stream_response() {
    let server = MockServer::start().await;
    let provider_id = "p-stream-fake";
    mount_chat_response(&server, "http.execute", "ok").await;
    let client = client(
        provider_id,
        &server,
        json!({"stream_behavior": "fake"}),
        ParamPolicy::RejectUnsupported,
    );

    let mut stream = client
        .chat()
        .create_stream(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();
    let chunk = stream.next().await.unwrap().unwrap();

    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("ok"));
    assert_eq!(request_count(&server).await, 1);
    assert_eq!(
        take_events(provider_id),
        vec![
            "supported_openai_params",
            "translate_messages",
            "map_openai_params",
            "validate_environment",
            "endpoint",
            "transform_request",
            "sign_request",
            "http.execute",
            "transform_response",
        ]
    );
}

#[tokio::test]
async fn create_stream_fake_lets_adapter_transform_non_success_status_into_business_error() {
    let server = MockServer::start().await;
    let provider_id = "p-stream-fake-error";
    mount_error_response(&server, "http.execute").await;
    let client = client(
        provider_id,
        &server,
        json!({"stream_behavior": "fake"}),
        ParamPolicy::RejectUnsupported,
    );

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
            "supported_openai_params",
            "translate_messages",
            "map_openai_params",
            "validate_environment",
            "endpoint",
            "transform_request",
            "sign_request",
            "http.execute",
            "transform_error_response",
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
        ParamPolicy::RejectUnsupported,
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
    let client = client(
        provider_id,
        &server,
        Value::Null,
        ParamPolicy::RejectUnsupported,
    );

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
            "supported_openai_params",
            "translate_messages",
            "map_openai_params",
            "validate_environment",
            "endpoint",
            "transform_request",
            "sign_request",
            "http.stream",
            "transform_error_response",
        ]
    );
}
