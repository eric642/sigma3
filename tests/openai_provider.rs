use std::collections::{BTreeMap, HashMap};

use futures_util::StreamExt;
use http::StatusCode;
use serde_json::{Value, json};
use sigma::types::chat::{
    ChatCompletionNamedToolChoice, ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestDeveloperMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartAudio, ChatCompletionRequestMessageContentPartFile,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionResponseMessageAnnotation,
    ChatCompletionStreamOptions, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionTools, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    CreateChatCompletionRequestParamsArgs, FileObject, InputAudio, InputAudioFormat,
    PredictionContent, PredictionContentContent, ServiceTier,
};
use sigma::types::shared::{
    FunctionName, FunctionObject, ImageUrl, ResponseFormat, ResponseFormatJsonSchema,
};
use sigma::{
    ChatParamConfig, ChatParameterMap, Client, ClientConfig, ModelDeploymentConfig, ModelName,
    ModelRef, ProviderCatalog, ProviderCommonConfig, ProviderConfigMap, ProviderId,
    ProviderInstanceConfig, ProviderKind, SecretString, SigmaError,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

fn openai_config(
    kind: &str,
    provider_id: &str,
    api_base: impl Into<Option<String>>,
    api_key: impl Into<Option<SecretString>>,
    headers: HashMap<String, String>,
    provider_config: Value,
) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: ProviderId::from(provider_id),
            kind: ProviderKind::from(kind),
            common: ProviderCommonConfig {
                api_base: api_base.into(),
                api_key: api_key.into(),
                headers,
                chat_params: ChatParamConfig::default(),
            },
            config: provider_config_map(provider_config),
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
    }
}

fn provider_config_map(value: Value) -> ProviderConfigMap {
    match value {
        Value::Object(map) => map,
        Value::Null => ProviderConfigMap::new(),
        other => panic!("provider config must be an object or null, got {other:?}"),
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

fn stream_chunk(id: &str, index: u32, delta: Value, finish_reason: Value) -> Value {
    json!({
        "id": id,
        "choices": [{
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
        "created": 1,
        "model": "gpt-4o-mini",
        "object": "chat.completion.chunk",
        "usage": null,
    })
}

async fn mount_stream_response(server: &MockServer, body: String) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(server)
        .await;
}

#[test]
fn catalog_from_inventory_collects_openai_provider_registrations() {
    let catalog = ProviderCatalog::from_inventory().unwrap();

    assert!(catalog.contains_kind(&ProviderKind::from("openai")));
    assert!(catalog.contains_kind(&ProviderKind::from("openai-compatible")));
}

#[test]
fn catalog_from_inventory_exposes_openai_provider_config_schemas() {
    let catalog = ProviderCatalog::from_inventory().unwrap();
    let schemas = catalog.provider_instance_config_schemas();
    let openai = schemas
        .iter()
        .find(|schema| schema.kind == "openai")
        .unwrap();
    let compatible = schemas
        .iter()
        .find(|schema| schema.kind == "openai-compatible")
        .unwrap();

    assert_eq!(
        openai.schema["properties"]["config"]["additionalProperties"],
        false
    );
    assert_eq!(
        openai.schema["properties"]["chat_params"]["properties"]["policy"]["default"],
        "reject_unsupported"
    );
    assert_eq!(
        compatible.schema["properties"]["config"]["additionalProperties"],
        false
    );
}

#[test]
fn openai_provider_rejects_unknown_provider_config_fields() {
    let err = match Client::build(openai_config(
        "openai",
        "openai-primary",
        Some("http://localhost:8080/v1".to_string()),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        json!({ "unexpected": true }),
    )) {
        Ok(_) => panic!("expected provider config error"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        SigmaError::ProviderConfig {
            provider: Some(provider),
            message
        } if provider == "openai-primary"
            && message.contains("unknown field")
    ));
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
async fn openai_create_preserves_developer_role_for_openai() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-developer",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let request = CreateChatCompletionRequestArgs::default()
        .messages(vec![
            ChatCompletionRequestMessage::Developer(ChatCompletionRequestDeveloperMessage {
                content: ChatCompletionRequestDeveloperMessageContent::Text(
                    "Be precise.".to_string(),
                ),
                name: None,
            }),
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage::from("hello")),
        ])
        .model(ModelRef::model("gpt-public"))
        .build()
        .unwrap();

    client.chat().create(&request).await.unwrap();

    assert_eq!(last_body(&server).await["messages"][0]["role"], "developer");
}

#[tokio::test]
async fn openai_create_sends_multimodal_user_content_parts() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-multimodal",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let request = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: "Describe these inputs.".to_string(),
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: "data:image/png;base64,AAAA".to_string(),
                                detail: None,
                            },
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::InputAudio(
                        ChatCompletionRequestMessageContentPartAudio {
                            input_audio: InputAudio {
                                data: "UklGRg==".to_string(),
                                format: InputAudioFormat::Wav,
                            },
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::File(
                        ChatCompletionRequestMessageContentPartFile {
                            file: FileObject {
                                file_data: Some("data:application/pdf;base64,JVBERi0=".to_string()),
                                file_id: None,
                                filename: Some("paper.pdf".to_string()),
                            },
                        },
                    ),
                ]),
                name: None,
            },
        )])
        .model(ModelRef::model("gpt-public"))
        .build()
        .unwrap();

    client.chat().create(&request).await.unwrap();

    assert_eq!(
        last_body(&server).await["messages"][0]["content"],
        json!([
            {"type": "text", "text": "Describe these inputs."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": null}},
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "paper.pdf"}},
        ])
    );
}

#[tokio::test]
async fn openai_create_passes_prediction_content() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-prediction",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.prediction = Some(PredictionContent::Content(PredictionContentContent::Text(
        "expected output".to_string(),
    )));

    client.chat().create(&request).await.unwrap();

    assert_eq!(
        last_body(&server).await["prediction"],
        json!({"type": "content", "content": "expected output"})
    );
}

#[tokio::test]
async fn openai_create_passes_safety_identifier() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-safety",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.safety_identifier = Some("user_123".to_string());

    client.chat().create(&request).await.unwrap();

    assert_eq!(last_body(&server).await["safety_identifier"], "user_123");
}

#[tokio::test]
async fn openai_create_passes_service_tier() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-service-tier",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.service_tier = Some(ServiceTier::Priority);

    client.chat().create(&request).await.unwrap();

    assert_eq!(last_body(&server).await["service_tier"], "priority");
}

#[tokio::test]
async fn openai_create_passes_response_format_json_schema() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-json-schema",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.response_format = Some(ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: Some("Weather response".to_string()),
            name: "weather_response".to_string(),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    });

    client.chat().create(&request).await.unwrap();

    assert_eq!(
        last_body(&server).await["response_format"],
        json!({
            "type": "json_schema",
            "json_schema": {
                "description": "Weather response",
                "name": "weather_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"],
                    "additionalProperties": false
                },
                "strict": true
            }
        })
    );
}

#[tokio::test]
async fn openai_create_passes_tools_tool_choice_and_parallel_tool_calls() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-tools",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.tools = Some(vec![ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            })),
            strict: Some(true),
        },
    })]);
    request.params.tool_choice = Some(ChatCompletionToolChoiceOption::Function(
        ChatCompletionNamedToolChoice {
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));
    request.params.parallel_tool_calls = Some(true);

    client.chat().create(&request).await.unwrap();

    assert_eq!(
        last_body(&server).await["tools"][0]["function"]["name"],
        "get_weather"
    );
}

#[tokio::test]
async fn openai_create_sends_configured_openai_organization_header() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let headers = HashMap::from([(
        "OpenAI-Organization".to_string(),
        "org_test_123".to_string(),
    )]);
    let client = Client::build(openai_config(
        "openai",
        "openai-org",
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

    let request = last_request(&server).await;
    assert_eq!(
        request
            .headers
            .get("openai-organization")
            .and_then(|value| value.to_str().ok()),
        Some("org_test_123")
    );
}

#[tokio::test]
async fn openai_create_rejects_sdk_transport_controls_as_chat_params() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let mut config = openai_config(
        "openai",
        "openai-sdk-controls",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    );
    config.deployments[0]
        .defaults
        .insert("max_retries".to_string(), json!(0));
    config.deployments[0]
        .defaults
        .insert("extra_headers".to_string(), json!({"x-test": "1"}));
    config.deployments[0]
        .defaults
        .insert("extra_body".to_string(), json!({"provider_native": true}));
    let client = Client::build(config).unwrap();

    let err = client
        .chat()
        .create(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap_err();

    match err {
        SigmaError::UnsupportedParams {
            provider,
            mut params,
        } => {
            params.sort();
            assert_eq!(provider, "openai-sdk-controls");
            assert_eq!(params, ["extra_body", "extra_headers", "max_retries"]);
        }
        other => panic!("expected unsupported params, got {other:?}"),
    }
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
async fn compatible_create_preserves_max_completion_tokens_by_default() {
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
    assert_eq!(body["max_completion_tokens"], 42);
    assert!(body.get("max_tokens").is_none());
}

#[tokio::test]
async fn compatible_create_maps_max_completion_tokens_when_chat_params_rename_configured() {
    let server = MockServer::start().await;
    mount_json_response(
        &server,
        "/v1/chat/completions",
        response_body("gpt-4o-mini", "ok"),
    )
    .await;
    let mut config = openai_config(
        "openai-compatible",
        "compatible-tokens",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("token")),
        HashMap::new(),
        Value::Null,
    );
    config.providers[0].common.chat_params.rename = Some(BTreeMap::from([(
        "max_completion_tokens".to_string(),
        "max_tokens".to_string(),
    )]));
    let client = Client::build(config).unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.max_completion_tokens = Some(42);

    client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(body["max_tokens"], 42);
    assert!(body.get("max_completion_tokens").is_none());
}

#[test]
fn compatible_provider_rejects_legacy_token_mapping_config() {
    let err = match Client::build(openai_config(
        "openai-compatible",
        "compatible-legacy-config",
        Some("http://localhost:8080/v1".to_string()),
        Some(SecretString::from("token")),
        HashMap::new(),
        json!({
            "map_max_completion_tokens_to_max_tokens": false
        }),
    )) {
        Ok(_) => panic!("expected provider config error"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        SigmaError::ProviderConfig {
            provider: Some(provider),
            message
        } if provider == "compatible-legacy-config"
            && message.contains("unknown field")
    ));
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
    mount_stream_response(
        &server,
        format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            stream_chunk(
                "chunk-1",
                0,
                json!({
                    "role": "assistant",
                    "content": "hel",
                }),
                Value::Null,
            ),
            stream_chunk(
                "chunk-1",
                0,
                json!({
                    "content": "lo",
                }),
                json!("stop"),
            )
        ),
    )
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

#[tokio::test]
async fn openai_create_parses_annotations_and_prediction_usage_details() {
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
                    "content": "See source.",
                    "annotations": [{
                        "type": "url_citation",
                        "url_citation": {
                            "start_index": 0,
                            "end_index": 10,
                            "title": "Source",
                            "url": "https://example.test/source"
                        }
                    }]
                },
                "finish_reason": "stop",
            }],
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 1,
                    "rejected_prediction_tokens": 0,
                    "reasoning_tokens": 0,
                    "audio_tokens": 0
                }
            },
        }),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-response-rich",
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

    let annotation = &response.choices[0].message.annotations.as_ref().unwrap()[0];
    assert!(matches!(
        annotation,
        ChatCompletionResponseMessageAnnotation::UrlCitation { url_citation }
            if url_citation.url == "https://example.test/source"
    ));
    assert_eq!(
        response
            .usage
            .unwrap()
            .completion_tokens_details
            .unwrap()
            .accepted_prediction_tokens,
        Some(1)
    );
}

#[tokio::test]
async fn openai_create_stream_preserves_usage_chunk_when_include_usage_true() {
    let server = MockServer::start().await;
    mount_stream_response(
        &server,
        format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            stream_chunk(
                "chunk-usage",
                0,
                json!({
                    "role": "assistant",
                    "content": "hi",
                }),
                Value::Null,
            ),
            json!({
                "id": "chunk-usage",
                "choices": [],
                "created": 1,
                "model": "gpt-4o-mini",
                "object": "chat.completion.chunk",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3
                },
            })
        ),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-stream-usage",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.stream_options = Some(ChatCompletionStreamOptions {
        include_usage: Some(true),
        include_obfuscation: None,
    });

    let chunks = client
        .chat()
        .create_stream(&request)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(
        chunks.last().unwrap().usage.as_ref().unwrap().total_tokens,
        3
    );
    assert_eq!(
        last_body(&server).await["stream_options"]["include_usage"],
        true
    );
}

#[tokio::test]
async fn openai_create_stream_preserves_choice_indices_for_n_greater_than_one() {
    let server = MockServer::start().await;
    mount_stream_response(
        &server,
        format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            stream_chunk("chunk-n", 0, json!({"content": "a"}), Value::Null),
            stream_chunk("chunk-n", 1, json!({"content": "b"}), json!("stop")),
        ),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-stream-n",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("gpt-public"));
    request.params.n = Some(2);

    let chunks = client
        .chat()
        .create_stream(&request)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let indices = chunks
        .iter()
        .map(|chunk| chunk.choices[0].index)
        .collect::<Vec<_>>();

    assert_eq!(indices, [0, 1]);
}

#[tokio::test]
async fn openai_create_stream_preserves_tool_call_chunks() {
    let server = MockServer::start().await;
    mount_stream_response(
        &server,
        format!(
            "data: {}\n\ndata: [DONE]\n\n",
            stream_chunk(
                "chunk-tool",
                0,
                json!({
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"SF\"}"
                        }
                    }]
                }),
                Value::Null,
            )
        ),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-stream-tool",
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

    let tool_call = &chunks[0].choices[0].delta.tool_calls.as_ref().unwrap()[0];
    assert_eq!(tool_call.id.as_deref(), Some("call_123"));
    assert_eq!(
        tool_call.function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
}

#[tokio::test]
async fn openai_create_stream_accepts_crlf_comments_and_raw_json_lines() {
    let server = MockServer::start().await;
    mount_stream_response(
        &server,
        format!(
            ": keepalive\r\ndata: {}\r\n\r\n{}\n[DONE]\n",
            stream_chunk("chunk-crlf", 0, json!({"content": "a"}), Value::Null),
            stream_chunk("chunk-crlf", 0, json!({"content": "b"}), json!("stop")),
        ),
    )
    .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-stream-crlf",
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
    let content = chunks
        .iter()
        .filter_map(|chunk| chunk.choices[0].delta.content.as_deref())
        .collect::<String>();

    assert_eq!(content, "ab");
}

#[tokio::test]
async fn openai_create_stream_surfaces_invalid_json_as_provider_response_error() {
    let server = MockServer::start().await;
    mount_stream_response(&server, "data: {\"bad\":\n\n".to_string()).await;
    let client = Client::build(openai_config(
        "openai",
        "openai-stream-invalid-json",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut stream = client
        .chat()
        .create_stream(&request(ModelRef::model("gpt-public")))
        .await
        .unwrap();

    let err = stream.next().await.unwrap().unwrap_err();

    assert!(matches!(
        err,
        SigmaError::ProviderResponse { provider, .. }
            if provider == "openai-stream-invalid-json"
    ));
}

#[tokio::test]
async fn openai_create_stream_maps_error_status_to_provider_business_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(StatusCode::TOO_MANY_REQUESTS.as_u16()).set_body_json(json!({
                "error": {
                    "message": "rate limited",
                    "type": "rate_limit_error",
                    "code": "rate_limit",
                },
            })),
        )
        .mount(&server)
        .await;
    let client = Client::build(openai_config(
        "openai",
        "openai-stream-error",
        Some(format!("{}/v1", server.uri())),
        Some(SecretString::from("sk-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

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
            ..
        } if provider == "openai-stream-error"
            && status == StatusCode::TOO_MANY_REQUESTS
            && code == "rate_limit"
            && message == "rate limited"
    ));
}
