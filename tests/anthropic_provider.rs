use std::collections::HashMap;

use futures_util::StreamExt;
use http::StatusCode;
use serde_json::{Value, json};
use sigma::types::chat::{
    CacheControl, ChatCompletionNamedToolChoice, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartFile, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPart,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionTools, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, CreateChatCompletionRequestParamsArgs, FileObject,
    FinishReason,
};
use sigma::types::shared::{FunctionName, FunctionObject, ImageUrl, ResponseFormat};
use sigma::{
    ChatParamConfig, ChatParameterMap, Client, ClientConfig, ModelDeploymentConfig, ModelName,
    ModelRef, ProviderCatalog, ProviderCommonConfig, ProviderConfigMap, ProviderId,
    ProviderInstanceConfig, ProviderKind, SecretString, SigmaError,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

fn anthropic_config(
    provider_id: &str,
    api_base: impl Into<Option<String>>,
    api_key: impl Into<Option<SecretString>>,
    headers: HashMap<String, String>,
    provider_config: Value,
) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: ProviderId::from(provider_id),
            kind: ProviderKind::from("anthropic"),
            common: ProviderCommonConfig {
                api_base: api_base.into(),
                api_key: api_key.into(),
                headers,
                chat_params: ChatParamConfig::default(),
            },
            config: provider_config_map(provider_config),
        }],
        deployments: vec![ModelDeploymentConfig {
            id: "claude-chat".into(),
            public_model: ModelName::from("claude-public"),
            provider: ProviderId::from(provider_id),
            provider_model: ModelName::from("claude-3-5-sonnet-20241022"),
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
                .max_tokens(128u32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

fn anthropic_response(content: Vec<Value>, stop_reason: &str) -> Value {
    json!({
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
            "output_tokens": 4
        }
    })
}

async fn mount_anthropic_response(server: &MockServer, body: Value) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
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
fn catalog_from_inventory_collects_anthropic_provider_registration() {
    let catalog = ProviderCatalog::from_inventory().unwrap();

    assert!(catalog.contains_kind(&ProviderKind::from("anthropic")));
}

#[test]
fn anthropic_provider_rejects_unknown_provider_config_fields() {
    let err = match Client::build(anthropic_config(
        "anthropic-primary",
        Some("http://localhost:8080".to_string()),
        Some(SecretString::from("sk-ant-test")),
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
        } if provider == "anthropic-primary" && message.contains("unknown field")
    ));
}

#[tokio::test]
async fn anthropic_create_posts_messages_body_and_headers() {
    let server = MockServer::start().await;
    mount_anthropic_response(
        &server,
        anthropic_response(vec![json!({"type": "text", "text": "ok"})], "end_turn"),
    )
    .await;
    let client = Client::build(anthropic_config(
        "anthropic-primary",
        Some(server.uri()),
        Some(SecretString::from("sk-ant-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let request = CreateChatCompletionRequestArgs::default()
        .messages(vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text("Be terse.".to_string()),
                name: None,
            }),
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestUserMessageContentPart::Text("Describe it.".into()),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        sigma::types::chat::ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: "data:image/png;base64,QUJD".to_string(),
                                detail: None,
                            },
                            cache_control: None,
                        },
                    ),
                ]),
                name: None,
            }),
        ])
        .model(ModelRef::model("claude-public"))
        .params(
            CreateChatCompletionRequestParamsArgs::default()
                .temperature(0.2f32)
                .max_completion_tokens(32u32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let response = client.chat().create(&request).await.unwrap();

    assert_eq!(response.choices[0].message.content.as_deref(), Some("ok"));
    let request = last_request(&server).await;
    assert_eq!(
        request
            .headers
            .get("x-api-key")
            .and_then(|value| value.to_str().ok()),
        Some("sk-ant-test")
    );
    assert_eq!(
        request
            .headers
            .get("anthropic-version")
            .and_then(|value| value.to_str().ok()),
        Some("2023-06-01")
    );

    let body: Value = request.body_json().unwrap();
    assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
    assert_eq!(body["max_tokens"], 32);
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "Be terse."}])
    );
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "Describe it."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
        ])
    );
}

#[tokio::test]
async fn anthropic_create_maps_content_part_cache_control() {
    let server = MockServer::start().await;
    mount_anthropic_response(
        &server,
        anthropic_response(vec![json!({"type": "text", "text": "ok"})], "end_turn"),
    )
    .await;
    let client = Client::build(anthropic_config(
        "anthropic-cache-control",
        Some(server.uri()),
        Some(SecretString::from("sk-ant-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let request = CreateChatCompletionRequestArgs::default()
        .messages(vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Array(vec![
                    ChatCompletionRequestSystemMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: "Use cached policy.".to_string(),
                            cache_control: Some(CacheControl::ephemeral()),
                        },
                    ),
                ]),
                name: None,
            }),
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: "Cached user text.".to_string(),
                            cache_control: Some(CacheControl::ephemeral()),
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: "data:image/png;base64,QUJD".to_string(),
                                detail: None,
                            },
                            cache_control: Some(CacheControl::ephemeral()),
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::File(
                        ChatCompletionRequestMessageContentPartFile {
                            file: FileObject {
                                file_id: Some("file_123".to_string()),
                                file_data: None,
                                filename: None,
                                format: None,
                                detail: None,
                                video_metadata: None,
                            },
                            cache_control: Some(CacheControl::ephemeral()),
                        },
                    ),
                ]),
                name: None,
            }),
        ])
        .model(ModelRef::model("claude-public"))
        .build()
        .unwrap();

    client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(
        body["system"],
        json!([
            {
                "type": "text",
                "text": "Use cached policy.",
                "cache_control": {"type": "ephemeral"}
            }
        ])
    );
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {
                "type": "text",
                "text": "Cached user text.",
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "document",
                "source": {"type": "file", "file_id": "file_123"},
                "cache_control": {"type": "ephemeral"}
            }
        ])
    );
}

#[tokio::test]
async fn anthropic_create_sends_native_provider_options_and_infers_beta_headers() {
    let server = MockServer::start().await;
    mount_anthropic_response(
        &server,
        anthropic_response(vec![json!({"type": "text", "text": "ok"})], "end_turn"),
    )
    .await;
    let provider_id = "anthropic-primary";
    let client = Client::build(anthropic_config(
        provider_id,
        Some(server.uri()),
        Some(SecretString::from("sk-ant-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("claude-public"));
    let context_management = json!({
        "edits": [{"type": "context_management_compact"}],
    });
    let mcp_servers = json!([{
        "type": "url",
        "name": "tools",
        "url": "https://example.com/mcp",
    }]);
    let container = json!({"type": "auto"});
    let output_format = json!({
        "type": "json_schema",
        "schema": {"type": "object", "properties": {}},
    });
    let output_config = json!({"effort": "low"});
    let mut options = ChatParameterMap::new();
    options.insert("context_management".to_string(), context_management.clone());
    options.insert("mcp_servers".to_string(), mcp_servers.clone());
    options.insert("container".to_string(), container.clone());
    options.insert("output_format".to_string(), output_format.clone());
    options.insert("output_config".to_string(), output_config.clone());
    options.insert("speed".to_string(), json!("fast"));
    options.insert(
        "anthropic_beta".to_string(),
        json!(["manual-beta-2026-01-01"]),
    );
    request
        .provider_options
        .insert(ProviderId::from(provider_id), options);

    client.chat().create(&request).await.unwrap();

    let request = last_request(&server).await;
    let body: Value = request.body_json().unwrap();
    assert_eq!(body["context_management"], context_management);
    assert_eq!(body["mcp_servers"], mcp_servers);
    assert_eq!(body["container"], container);
    assert_eq!(body["output_format"], output_format);
    assert_eq!(body["output_config"], output_config);
    assert_eq!(body["speed"], "fast");
    assert!(body.get("anthropic_beta").is_none());

    let beta_header = request
        .headers
        .get("anthropic-beta")
        .and_then(|value| value.to_str().ok())
        .unwrap();
    for expected in [
        "compact-2026-01-12",
        "context-management-2025-06-27",
        "fast-mode-2026-02-01",
        "manual-beta-2026-01-01",
        "mcp-client-2025-04-04",
        "structured-outputs-2025-11-13",
    ] {
        assert!(
            beta_header.split(',').any(|value| value == expected),
            "missing {expected} in {beta_header}"
        );
    }
}

#[tokio::test]
async fn anthropic_provider_options_output_format_overrides_portable_response_format() {
    let server = MockServer::start().await;
    mount_anthropic_response(
        &server,
        anthropic_response(vec![json!({"type": "text", "text": "ok"})], "end_turn"),
    )
    .await;
    let provider_id = "anthropic-primary";
    let client = Client::build(anthropic_config(
        provider_id,
        Some(server.uri()),
        Some(SecretString::from("sk-ant-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("claude-public"));
    request.params.response_format = Some(ResponseFormat::JsonObject);
    let output_format = json!({
        "type": "json_schema",
        "schema": {"type": "object", "additionalProperties": false},
    });
    let mut options = ChatParameterMap::new();
    options.insert("output_format".to_string(), output_format.clone());
    request
        .provider_options
        .insert(ProviderId::from(provider_id), options);

    client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(body["output_format"], output_format);
    assert!(body.get("response_format").is_none());
    assert!(body.get("tools").is_none());
}

#[tokio::test]
async fn anthropic_create_maps_tools_tool_choice_and_reverses_sanitized_tool_names() {
    let server = MockServer::start().await;
    mount_anthropic_response(
        &server,
        anthropic_response(
            vec![json!({
                "type": "tool_use",
                "id": "toolu_123",
                "name": "weather_lookup",
                "input": {"city": "SF"}
            })],
            "tool_use",
        ),
    )
    .await;
    let client = Client::build(anthropic_config(
        "anthropic-tools",
        Some(server.uri()),
        Some(SecretString::from("sk-ant-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("claude-public"));
    request.params.tools = Some(vec![ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "weather.lookup".to_string(),
            description: Some("Get weather".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
            strict: Some(true),
        },
    })]);
    request.params.tool_choice = Some(ChatCompletionToolChoiceOption::Function(
        ChatCompletionNamedToolChoice {
            function: FunctionName {
                name: "weather.lookup".to_string(),
            },
        },
    ));
    request.params.parallel_tool_calls = Some(false);

    let response = client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(body["tools"][0]["name"], "weather_lookup");
    assert_eq!(body["tool_choice"]["name"], "weather_lookup");
    assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
    let tool_call = &response.choices[0].message.tool_calls.as_ref().unwrap()[0];
    assert_eq!(
        serde_json::to_value(tool_call).unwrap()["function"]["name"],
        "weather.lookup"
    );
}

#[tokio::test]
async fn anthropic_create_maps_response_format_reasoning_and_usage() {
    let server = MockServer::start().await;
    mount_anthropic_response(
        &server,
        anthropic_response(
            vec![
                json!({"type": "thinking", "thinking": "I will answer.", "signature": "sig"}),
                json!({"type": "text", "text": "{\"answer\":true}", "citations": [{
                    "type": "char_location",
                    "cited_text": "source",
                    "document_title": "Doc",
                    "start_char_index": 0,
                    "end_char_index": 6
                }]}),
            ],
            "end_turn",
        ),
    )
    .await;
    let client = Client::build(anthropic_config(
        "anthropic-json",
        Some(server.uri()),
        Some(SecretString::from("sk-ant-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();
    let mut request = request(ModelRef::model("claude-public"));
    request.params.response_format = Some(ResponseFormat::JsonObject);
    request.params.reasoning_effort = Some(sigma::types::shared::ReasoningEffort::Low);

    let response = client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(body["thinking"]["type"], "enabled");
    assert_eq!(body["tools"][0]["name"], "json_tool_call");
    assert_eq!(response.usage.as_ref().unwrap().prompt_tokens, 15);
    assert_eq!(response.usage.as_ref().unwrap().completion_tokens, 4);
    assert_eq!(
        response.choices[0]
            .message
            .thinking_blocks
            .as_ref()
            .unwrap()[0]
            .thinking
            .as_deref(),
        Some("I will answer.")
    );
    assert!(response.choices[0].message.annotations.is_some());
}

#[tokio::test]
async fn anthropic_create_maps_error_body_to_provider_business_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(StatusCode::TOO_MANY_REQUESTS.as_u16()).set_body_json(json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "rate limited"
                }
            })),
        )
        .mount(&server)
        .await;
    let client = Client::build(anthropic_config(
        "anthropic-error",
        Some(server.uri()),
        Some(SecretString::from("sk-ant-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

    let err = client
        .chat()
        .create(&request(ModelRef::model("claude-public")))
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        SigmaError::ProviderBusiness {
            provider,
            status,
            code: Some(code),
            message,
            ..
        } if provider == "anthropic-error"
            && status == StatusCode::TOO_MANY_REQUESTS
            && code == "rate_limit_error"
            && message == "rate limited"
    ));
}

#[tokio::test]
async fn anthropic_create_stream_parses_text_tool_and_usage_events() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-sonnet-20241022\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_stream\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"SF\\\"}\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;
    let client = Client::build(anthropic_config(
        "anthropic-stream",
        Some(server.uri()),
        Some(SecretString::from("sk-ant-test")),
        HashMap::new(),
        Value::Null,
    ))
    .unwrap();

    let chunks = client
        .chat()
        .create_stream(&request(ModelRef::model("claude-public")))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("hel"));
    assert_eq!(
        chunks
            .iter()
            .find_map(|chunk| chunk.choices[0].delta.tool_calls.as_ref())
            .unwrap()[0]
            .id
            .as_deref(),
        Some("toolu_stream")
    );
    assert_eq!(
        chunks.last().unwrap().choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    assert_eq!(
        chunks.last().unwrap().usage.as_ref().unwrap().total_tokens,
        8
    );
    assert_eq!(last_body(&server).await["stream"], true);
}
