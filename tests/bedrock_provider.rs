use std::collections::HashMap;

use futures_util::StreamExt;
use serde_json::{Value, json};
use sigma::types::chat::{
    ChatRequest, ChatRequestParams, FinishReason, FunctionTool, StreamOptions, SystemMessage,
    TextContent, ToolCall, ToolChoice, ToolDefinition, UserMessage, WebSearchOptions,
};
use sigma::types::shared::{FunctionObject, ReasoningEffort, ResponseFormat};
use sigma::{
    Client, ClientConfig, ModelDeploymentConfig, ModelName, ModelRef, ProviderCatalog,
    ProviderCommonConfig, ProviderConfigMap, ProviderId, ProviderInstanceConfig, ProviderKind,
    SigmaError,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

fn bedrock_config(api_base: String, provider_config: Value) -> ClientConfig {
    bedrock_config_with_provider_model(
        api_base,
        provider_config,
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
}

fn bedrock_config_with_provider_model(
    api_base: String,
    provider_config: Value,
    provider_model: &str,
) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: ProviderId::from("bedrock-primary"),
            kind: ProviderKind::from("bedrock"),
            common: ProviderCommonConfig {
                api_base: Some(api_base),
                api_key: None,
                headers: HashMap::new(),
            },
            config: provider_config_map(provider_config),
        }],
        deployments: vec![ModelDeploymentConfig {
            id: "bedrock-chat".into(),
            public_model: ModelName::from("bedrock-public"),
            provider: ProviderId::from("bedrock-primary"),
            provider_model: ModelName::from(provider_model),
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

fn static_aws_config() -> Value {
    json!({
        "region": "us-east-1",
        "access_key_id": "AKIA_TEST",
        "secret_access_key": "bedrock-test-secret",
        "session_token": "bedrock-session-token"
    })
}

fn static_aws_config_without_region() -> Value {
    json!({
        "access_key_id": "AKIA_TEST",
        "secret_access_key": "bedrock-test-secret"
    })
}

fn converse_response(content: &str) -> Value {
    json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": content}]
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 4,
            "totalTokens": 14,
            "cacheReadInputTokens": 2,
            "cacheWriteInputTokens": 3
        },
        "metrics": {"latencyMs": 12}
    })
}

async fn mount_bedrock_response(server: &MockServer, body: Value) {
    Mock::given(method("POST"))
        .and(path(
            "/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

async fn mount_bedrock_response_for_path(server: &MockServer, request_path: &str, body: Value) {
    Mock::given(method("POST"))
        .and(path(request_path))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

async fn mount_bedrock_stream(server: &MockServer, body: Vec<u8>) {
    Mock::given(method("POST"))
        .and(path(
            "/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse-stream",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
        .mount(server)
        .await;
}

fn bedrock_event_stream(events: Vec<Value>) -> Vec<u8> {
    events
        .into_iter()
        .flat_map(|event| bedrock_event_stream_message(event.to_string().as_bytes()))
        .collect()
}

fn bedrock_event_stream_message(payload: &[u8]) -> Vec<u8> {
    let headers = event_stream_headers(&[
        (":message-type", "event"),
        (":event-type", "chunk"),
        (":content-type", "application/json"),
    ]);
    let total_len = 12 + headers.len() + payload.len() + 4;
    let mut message = Vec::new();
    message.extend_from_slice(&(total_len as u32).to_be_bytes());
    message.extend_from_slice(&(headers.len() as u32).to_be_bytes());
    let prelude_crc = crc32fast::hash(&message);
    message.extend_from_slice(&prelude_crc.to_be_bytes());
    message.extend_from_slice(&headers);
    message.extend_from_slice(payload);
    let message_crc = crc32fast::hash(&message);
    message.extend_from_slice(&message_crc.to_be_bytes());
    message
}

fn event_stream_headers(headers: &[(&str, &str)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (name, value) in headers {
        bytes.push(name.len() as u8);
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(7);
        bytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    bytes
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
fn catalog_from_inventory_collects_bedrock_provider_registration() {
    let catalog = ProviderCatalog::from_inventory().unwrap();

    assert!(catalog.contains_kind(&ProviderKind::from("bedrock")));
}

#[test]
fn bedrock_provider_rejects_unknown_provider_config_fields() {
    let err = match Client::build(bedrock_config(
        "http://localhost:8080".to_string(),
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
        } if provider == "bedrock-primary" && message.contains("unknown field")
    ));
}

#[tokio::test]
async fn bedrock_create_posts_converse_body_and_sigv4_headers() {
    let server = MockServer::start().await;
    mount_bedrock_response(&server, converse_response("ok")).await;
    let client = Client::build(bedrock_config(server.uri(), static_aws_config())).unwrap();
    let request = ChatRequest::new(
        ModelRef::model("bedrock-public"),
        vec![
            SystemMessage {
                content: TextContent::Text("Be terse.".to_string()),
                name: None,
            }
            .into(),
            UserMessage::from("hello").into(),
        ],
    )
    .with_params(ChatRequestParams {
        max_completion_tokens: Some(32),
        temperature: Some(0.2f32),
        ..Default::default()
    });

    let response = client.create(&request).await.unwrap();

    assert_eq!(response.choices[0].message.content.as_deref(), Some("ok"));
    assert_eq!(response.usage.as_ref().unwrap().prompt_tokens, 15);
    assert_eq!(response.usage.as_ref().unwrap().completion_tokens, 4);
    let request = last_request(&server).await;
    assert_eq!(
        request
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert_eq!(
        request
            .headers
            .get("x-amz-security-token")
            .and_then(|value| value.to_str().ok()),
        Some("bedrock-session-token")
    );
    let authorization = request
        .headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap();
    assert!(authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIA_TEST/"));
    assert!(authorization.contains("/us-east-1/bedrock/aws4_request"));

    let body: Value = request.body_json().unwrap();
    assert_eq!(body["system"], json!([{"text": "Be terse."}]));
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": [{"text": "hello"}]}])
    );
    assert_eq!(body["inferenceConfig"]["maxTokens"], 32);
    let temperature = body["inferenceConfig"]["temperature"].as_f64().unwrap();
    assert!((temperature - 0.2).abs() < 0.00001);
    assert!(body.get("model").is_none());
}

#[tokio::test]
async fn bedrock_create_maps_tools_provider_options_and_tool_use_response() {
    let server = MockServer::start().await;
    mount_bedrock_response(
        &server,
        json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{
                        "toolUse": {
                            "toolUseId": "tooluse_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"}
                        }
                    }]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}
        }),
    )
    .await;
    let client = Client::build(bedrock_config(server.uri(), static_aws_config())).unwrap();
    let request = ChatRequest::new(
        ModelRef::model("bedrock-public"),
        vec![UserMessage::from("use the tool").into()],
    )
    .with_params(ChatRequestParams {
        tools: Some(vec![ToolDefinition::Function(FunctionTool {
            function: FunctionObject {
                name: "get-weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                })),
                strict: Some(true),
            },
        })]),
        tool_choice: Some(ToolChoice::Function("get-weather".into())),
        ..Default::default()
    })
    .with_provider_option(
        ProviderId::from("bedrock-primary"),
        "guardrailConfig",
        json!({"guardrailIdentifier": "guardrail-id", "guardrailVersion": "DRAFT"}),
    );

    let response = client.create(&request).await.unwrap();

    assert_eq!(
        response.choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    let tool_call = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap()
        .first()
        .unwrap();
    let ToolCall::Function(tool_call) = tool_call else {
        panic!("expected function tool call");
    };
    assert_eq!(tool_call.id, "tooluse_1");
    assert_eq!(tool_call.function.name, "get-weather");
    assert_eq!(tool_call.function.arguments, r#"{"city":"Paris"}"#);

    let body = last_body(&server).await;
    assert_eq!(
        body["guardrailConfig"],
        json!({"guardrailIdentifier": "guardrail-id", "guardrailVersion": "DRAFT"})
    );
    assert_eq!(
        body["toolConfig"]["tools"][0]["toolSpec"]["name"],
        "get_weather"
    );
    assert_eq!(
        body["toolConfig"]["tools"][0]["toolSpec"]["description"],
        "Get weather"
    );
    let input_schema = &body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"];
    assert_eq!(input_schema["type"], "object");
    assert!(input_schema.get("$schema").is_none());
    assert!(input_schema.get("additionalProperties").is_none());
    assert!(input_schema.get("strict").is_none());
    assert_eq!(
        body["toolConfig"]["toolChoice"],
        json!({"tool": {"name": "get_weather"}})
    );
}

#[tokio::test]
async fn bedrock_create_uses_arn_region_for_sigv4_and_percent_encoded_model_path() {
    let server = MockServer::start().await;
    let provider_model =
        "arn:aws:bedrock:eu-central-1:000000000000:application-inference-profile/a0a0";
    mount_bedrock_response_for_path(
        &server,
        "/model/arn%3Aaws%3Abedrock%3Aeu-central-1%3A000000000000%3Aapplication-inference-profile%2Fa0a0/converse",
        converse_response("ok"),
    )
    .await;
    let client = Client::build(bedrock_config_with_provider_model(
        server.uri(),
        static_aws_config_without_region(),
        provider_model,
    ))
    .unwrap();

    let response = client
        .create(&ChatRequest::new(
            ModelRef::model("bedrock-public"),
            vec![UserMessage::from("hello").into()],
        ))
        .await
        .unwrap();

    assert_eq!(response.choices[0].message.content.as_deref(), Some("ok"));
    let request = last_request(&server).await;
    let authorization = request
        .headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap();
    assert!(authorization.contains("/eu-central-1/bedrock/aws4_request"));
}

#[tokio::test]
async fn bedrock_create_maps_gpt_oss_reasoning_and_json_response_format() {
    let server = MockServer::start().await;
    mount_bedrock_response_for_path(
        &server,
        "/model/openai.gpt-oss-20b-1%3A0/converse",
        converse_response(r#"{"ok":true}"#),
    )
    .await;
    let client = Client::build(bedrock_config_with_provider_model(
        server.uri(),
        static_aws_config(),
        "openai.gpt-oss-20b-1:0",
    ))
    .unwrap();
    let request = ChatRequest::new(
        ModelRef::model("bedrock-public"),
        vec![UserMessage::from("return json").into()],
    )
    .with_params(ChatRequestParams {
        reasoning_effort: Some(ReasoningEffort::Low),
        response_format: Some(ResponseFormat::JsonObject),
        ..Default::default()
    });

    let response = client.create(&request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some(r#"{"ok":true}"#)
    );
    let body = last_body(&server).await;
    assert_eq!(
        body["additionalModelRequestFields"]["reasoning_effort"],
        "low"
    );
    assert_eq!(
        body["toolConfig"]["tools"][0]["toolSpec"]["name"],
        "json_tool_call"
    );
    assert_eq!(
        body["toolConfig"]["toolChoice"],
        json!({"tool": {"name": "json_tool_call"}})
    );
}

#[tokio::test]
async fn bedrock_create_maps_nova_web_search_to_grounding_tool() {
    let server = MockServer::start().await;
    mount_bedrock_response_for_path(
        &server,
        "/model/us.amazon.nova-pro-v1%3A0/converse",
        converse_response("ok"),
    )
    .await;
    let client = Client::build(bedrock_config_with_provider_model(
        server.uri(),
        static_aws_config(),
        "us.amazon.nova-pro-v1:0",
    ))
    .unwrap();
    let request = ChatRequest::new(
        ModelRef::model("bedrock-public"),
        vec![UserMessage::from("search").into()],
    )
    .with_params(ChatRequestParams {
        web_search_options: Some(WebSearchOptions::default()),
        ..Default::default()
    });

    let response = client.create(&request).await.unwrap();

    assert_eq!(response.choices[0].message.content.as_deref(), Some("ok"));
    let body = last_body(&server).await;
    assert_eq!(
        body["toolConfig"]["tools"][0],
        json!({"systemTool": {"name": "nova_grounding"}})
    );
}

#[tokio::test]
async fn bedrock_create_stream_decodes_event_stream_text_tool_finish_and_usage() {
    let server = MockServer::start().await;
    mount_bedrock_stream(
        &server,
        bedrock_event_stream(vec![
            json!({"messageStart": {"role": "assistant"}}),
            json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hel"}}}),
            json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "lo"}}}),
            json!({"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather"}}}}),
            json!({"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"city\""}}}}),
            json!({"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ":\"Paris\"}"}}}}),
            json!({"messageStop": {"stopReason": "tool_use"}}),
            json!({"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8}}}),
        ]),
    )
    .await;
    let client = Client::build(bedrock_config(server.uri(), static_aws_config())).unwrap();
    let request = ChatRequest::new(
        ModelRef::model("bedrock-public"),
        vec![UserMessage::from("stream").into()],
    );

    let chunks = client
        .create_stream(&request)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("hel"));
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("lo"));
    let tool_start = chunks[2].choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap()
        .first()
        .unwrap();
    assert_eq!(tool_start.id.as_deref(), Some("tooluse_1"));
    assert_eq!(
        tool_start.function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        chunks[3].choices[0].delta.tool_calls.as_ref().unwrap()[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some("{\"city\"")
    );
    assert_eq!(
        chunks[5].choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    assert_eq!(
        chunks.last().unwrap().usage.as_ref().unwrap().total_tokens,
        8
    );

    let body = last_body(&server).await;
    if let Some(additional) = body
        .get("additionalModelRequestFields")
        .and_then(Value::as_object)
    {
        assert!(!additional.contains_key("stream_options"));
        assert!(!additional.contains_key("parallel_tool_calls"));
    }
}

#[tokio::test]
async fn bedrock_disambiguates_colliding_sanitized_tool_names() {
    let server = MockServer::start().await;
    mount_bedrock_response(
        &server,
        json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tu_a",
                                "name": "actions_foo",
                                "input": {"v": "a"}
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "tu_b",
                                "name": "actions_foo_2",
                                "input": {"v": "b"}
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
        }),
    )
    .await;
    let client = Client::build(bedrock_config(server.uri(), static_aws_config())).unwrap();
    let request = ChatRequest::new(
        ModelRef::model("bedrock-public"),
        vec![UserMessage::from("hi").into()],
    )
    .with_params(ChatRequestParams {
        tools: Some(vec![
            ToolDefinition::Function(FunctionTool {
                function: FunctionObject {
                    name: "actions/foo".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                },
            }),
            ToolDefinition::Function(FunctionTool {
                function: FunctionObject {
                    name: "actions.foo".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                },
            }),
        ]),
        ..Default::default()
    });

    let response = client.create(&request).await.unwrap();
    let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
    let names = tool_calls
        .iter()
        .map(|call| match call {
            ToolCall::Function(call) => call.function.name.clone(),
            other => panic!("unexpected tool call: {other:?}"),
        })
        .collect::<Vec<_>>();
    let expected: Vec<String> = vec!["actions/foo".into(), "actions.foo".into()];
    assert_eq!(names, expected, "reverse map must restore both originals");

    let body = last_body(&server).await;
    let tool_specs = body["toolConfig"]["tools"].as_array().unwrap();
    let wire_names = tool_specs
        .iter()
        .filter_map(|tool| {
            tool.get("toolSpec")
                .and_then(|spec| spec.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    let unique = wire_names.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(
        wire_names.len(),
        unique.len(),
        "wire tool names must be unique even when sanitized forms collide"
    );
}

#[tokio::test]
async fn bedrock_rejects_parallel_tool_calls_param() {
    let client = Client::build(bedrock_config(
        "http://localhost:8080".to_string(),
        static_aws_config(),
    ))
    .unwrap();
    let request = ChatRequest::new(
        ModelRef::model("bedrock-public"),
        vec![UserMessage::from("hi").into()],
    )
    .with_params(ChatRequestParams {
        parallel_tool_calls: Some(false),
        ..Default::default()
    });

    let err = client
        .create(&request)
        .await
        .expect_err("parallel_tool_calls is not supported by the Bedrock Converse API");
    assert!(matches!(
        err,
        SigmaError::UnsupportedParams { ref params, .. }
            if params.iter().any(|p| p == "parallel_tool_calls")
    ));
}

#[tokio::test]
async fn bedrock_rejects_stream_options_param() {
    let client = Client::build(bedrock_config(
        "http://localhost:8080".to_string(),
        static_aws_config(),
    ))
    .unwrap();
    let request = ChatRequest::new(
        ModelRef::model("bedrock-public"),
        vec![UserMessage::from("hi").into()],
    )
    .with_params(ChatRequestParams {
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        }),
        ..Default::default()
    });

    let err = client
        .create(&request)
        .await
        .expect_err("stream_options is not supported by the Bedrock Converse API");
    assert!(matches!(
        err,
        SigmaError::UnsupportedParams { ref params, .. }
            if params.iter().any(|p| p == "stream_options")
    ));
}
