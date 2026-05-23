use std::collections::HashMap;

use futures_util::StreamExt;
use http::StatusCode;
use serde_json::{Value, json};
use sigma::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, CreateChatCompletionRequestParamsArgs,
};
use sigma::{
    ChatParameterMap, Client, ClientConfig, ModelDeploymentConfig, ModelName, ModelRef,
    ParamPolicy, ProviderCatalog, ProviderId, ProviderInstanceConfig, ProviderKind, SecretString,
    SigmaError,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

fn openai_config(
    kind: &str,
    provider_id: &str,
    api_base: impl Into<Option<String>>,
    api_key: impl Into<Option<SecretString>>,
    headers: HashMap<String, String>,
    options: Value,
) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: ProviderId::from(provider_id),
            kind: ProviderKind::from(kind),
            api_base: api_base.into(),
            api_key: api_key.into(),
            headers,
            options,
        }],
        deployments: vec![ModelDeploymentConfig {
            id: "openai-chat".into(),
            public_model: ModelName::from("gpt-public"),
            provider: ProviderId::from(provider_id),
            provider_model: ModelName::from("gpt-4o-mini"),
            defaults: serde_json::Map::new(),
            model_info: Value::Null,
        }],
        default_model: None,
        param_policy: ParamPolicy::RejectUnsupported,
    }
}

fn request(model: ModelRef) -> CreateChatCompletionRequest {
    CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage::from("hello"),
        )])
        .model(model)
        .params(
            CreateChatCompletionRequestParamsArgs::default()
                .temperature(0.2f32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

fn response_body(model: &str, content: &str) -> Value {
    json!({
        "id": "chatcmpl_openai",
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
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    })
}

async fn mount_json_response(server: &MockServer, path_value: &'static str, body: Value) {
    Mock::given(method("POST"))
        .and(path(path_value))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

async fn last_request(server: &MockServer) -> WiremockRequest {
    server
        .received_requests()
        .await
        .unwrap()
        .last()
        .unwrap()
        .clone()
}

async fn last_body(server: &MockServer) -> Value {
    last_request(server).await.body_json().unwrap()
}

#[test]
fn catalog_from_inventory_collects_openai_provider_registrations() {
    let catalog = ProviderCatalog::from_inventory().unwrap();

    assert!(catalog.contains_kind(&ProviderKind::from("openai")));
    assert!(catalog.contains_kind(&ProviderKind::from("openai-compatible")));
}

#[tokio::test]
async fn openai_create_posts_openai_chat_completion_body() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-primary",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

    let response = client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    assert_eq!(response.choices[0].message.content.as_deref(), Some("ok"));
    let body = last_body(&server).await;
    assert_eq!(body["model"], "gpt-4o-mini");
    assert_eq!(
        body["messages"],
        json!([{
            "role": "user",
            "content": "hello",
        }])
    );
    assert!((body["temperature"].as_f64().unwrap() - 0.2).abs() < 0.000001);
}

#[tokio::test]
async fn openai_create_adds_bearer_auth_and_json_content_type() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-auth",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

    client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    let request = last_request(&server).await;
    assert_eq!(
        request
            .headers
            .get("authorization")
            .and_then(|value| value.to_str().ok()),
        Some("Bearer sk-test")
    );
    assert_eq!(
        request
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
}

#[tokio::test]
async fn openai_create_preserves_configured_authorization_header() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let headers = HashMap::from([(
        "Authorization".to_string(),
        "Bearer configured-token".to_string(),
    )]);
    let client = Client::build(openai_config(
        "openai",
        "openai-configured-auth",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        headers,
        Value::Null,
    ))
    .unwrap();

    client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    assert_eq!(
        last_request(&server)
            .await
            .headers
            .get("authorization")
            .and_then(|value| value.to_str().ok()),
        Some("Bearer configured-token")
    );
}

#[tokio::test]
async fn compatible_create_allows_missing_api_key_and_appends_chat_completions() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai-compatible",
        "compatible-local",
        Some(format!("{}/v1", server.uri())),
        None,
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

    client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    assert!(
        last_request(&server)
            .await
            .headers
            .get("authorization")
            .is_none()
    );
}

#[tokio::test]
async fn compatible_create_does_not_append_chat_completions_twice() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/custom/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai-compatible",
        "compatible-custom",
        Some(format!("{}/custom/chat/completions", server.uri())),
        Some(SecretString::from("token")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

    client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    assert_eq!(last_body(&server).await["model"], "gpt-4o-mini");
}

#[tokio::test]
async fn compatible_create_sends_provider_metadata_from_provider_override() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let provider_id = "zhipu";
    let client = Client::build(openai_config(
        "openai-compatible",
        provider_id,
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("token")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    let mut overrides = ChatParameterMap::new();
    overrides.insert("metadata".to_string(), json!({"trace_id": "trace-123"}));
    request
        .metadata
        .insert(ProviderId::from(provider_id), overrides);

    client.chat().create(&request).await.unwrap();

    assert_eq!(
        last_body(&server).await["metadata"],
        json!({"trace_id": "trace-123"})
    );
}

#[tokio::test]
async fn compatible_create_maps_max_completion_tokens_to_max_tokens_by_default() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai-compatible",
        "compatible-tokens",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("token")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.max_completion_tokens = Some(42);

    client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(body["max_tokens"], 42);
    assert!(body.get("max_completion_tokens").is_none());
}

#[tokio::test]
async fn compatible_create_sanitizes_null_usage_token_counts() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        json!({
            "id": "chatcmpl_openai",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok",
                },
                "finish_reason": "stop",
            }],
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": null,
                "completion_tokens": null,
                "total_tokens": null,
            },
        }),
    )
    .await;
    let client = Client::build(openai_config(
        "openai-compatible",
        "compatible-usage",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("token")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

    let response = client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    assert_eq!(response.usage.unwrap().total_tokens, 0);
}

#[tokio::test]
async fn create_maps_openai_error_body_to_provider_business_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(StatusCode::TOO_MANY_REQUESTS.as_u16()).set_body_json(json!({
                "error": {
                    "message": "rate limited",
                    "type": "rate_limit_error",
                    "code": "rate_limit",
                    "param": "messages",
                },
            })),
        )
        .mount(&server)
        .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-error",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

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
        } if provider == "openai-error"
            && status == StatusCode::TOO_MANY_REQUESTS
            && code == "rate_limit"
            && message == "rate limited"
            && details["type"] == "rate_limit_error"
            && details["param"] == "messages"
    ));
}

#[tokio::test]
async fn create_stream_parses_openai_sse_frames_and_done_marker() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({
                "id": "chunk-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "hel",
                    },
                    "finish_reason": null,
                }],
                "created": 1,
                "model": "gpt-4o-mini",
                "object": "chat.completion.chunk",
                "usage": null,
            }),
            json!({
                "id": "chunk-1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "lo",
                    },
                    "finish_reason": "stop",
                }],
                "created": 1,
                "model": "gpt-4o-mini",
                "object": "chat.completion.chunk",
                "usage": null,
            })
        )))
        .mount(&server)
        .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-stream",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

    let chunks = client
        .chat()
        .create_stream(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("hel"));
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("lo"));
    assert_eq!(last_body(&server).await["stream"], true);
}
