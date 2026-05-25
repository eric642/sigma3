use std::collections::HashMap;

use futures_util::StreamExt;
use http::StatusCode;
use serde_json::{Value, json};
use sigma::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionNamedToolChoice, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartAudio, ChatCompletionRequestMessageContentPartFile,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionTools, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, CreateChatCompletionRequestParamsArgs, FileObject,
    FinishReason, InputAudio, InputAudioFormat, StopConfiguration,
};
use sigma::types::shared::{
    FunctionName, FunctionObject, ImageDetail, ImageUrl, ReasoningEffort, ResponseFormat,
    ResponseFormatJsonSchema,
};
use sigma::{
    ChatParamConfig, Client, ClientConfig, ModelDeploymentConfig, ModelName, ModelRef,
    ProviderCatalog, ProviderCommonConfig, ProviderConfigMap, ProviderId, ProviderInstanceConfig,
    ProviderKind, SecretString, SigmaError,
};
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

fn gemini_config(api_base: String) -> ClientConfig {
    gemini_config_with_provider_model(api_base, "gemini-2.5-flash")
}

fn gemini_config_with_provider_model(api_base: String, provider_model: &str) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: ProviderId::from("gemini-primary"),
            kind: ProviderKind::from("gemini"),
            common: ProviderCommonConfig {
                api_base: Some(api_base),
                api_key: Some(SecretString::from("gemini-test-key")),
                headers: HashMap::new(),
                chat_params: ChatParamConfig::default(),
            },
            config: ProviderConfigMap::new(),
        }],
        deployments: vec![ModelDeploymentConfig {
            id: "gemini-chat".into(),
            public_model: ModelName::from("gemini-public"),
            provider: ProviderId::from("gemini-primary"),
            provider_model: ModelName::from(provider_model),
            defaults: serde_json::Map::new(),
            model_info: Value::Null,
        }],
        default_model: None,
    }
}

fn gemini_response(content: &str) -> Value {
    json!({
        "responseId": "gemini-response",
        "candidates": [{
            "index": 0,
            "content": {
                "role": "model",
                "parts": [{"text": content}]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 2,
            "totalTokenCount": 3,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}]
        }
    })
}

async fn mount_gemini_json(server: &MockServer, body: Value) {
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

async fn mount_gemini_stream(server: &MockServer, body: String) {
    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        ))
        .and(query_param("alt", "sse"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
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

fn basic_request() -> CreateChatCompletionRequest {
    CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage::from("hello"),
        )])
        .model(ModelRef::model("gemini-public"))
        .params(
            CreateChatCompletionRequestParamsArgs::default()
                .max_completion_tokens(32u32)
                .temperature(0.2f32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

#[test]
fn catalog_from_inventory_collects_gemini_provider_registration() {
    let catalog = ProviderCatalog::from_inventory().unwrap();

    assert!(catalog.contains_kind(&ProviderKind::from("gemini")));
}

#[tokio::test]
async fn gemini_create_uses_v1alpha_for_gemini_3_models() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1alpha/models/gemini-3-pro:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(gemini_response("ok")))
        .mount(&server)
        .await;
    let client = Client::build(gemini_config_with_provider_model(
        server.uri(),
        "gemini-3-pro",
    ))
    .unwrap();

    let response = client.chat().create(&basic_request()).await.unwrap();

    assert_eq!(response.choices[0].message.content.as_deref(), Some("ok"));
}

#[tokio::test]
async fn gemini_create_posts_generate_content_body_and_headers() {
    let server = MockServer::start().await;
    mount_gemini_json(&server, gemini_response("ok")).await;
    let client = Client::build(gemini_config(server.uri())).unwrap();
    let request = CreateChatCompletionRequestArgs::default()
        .messages(vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text("Be concise.".to_string()),
                name: None,
            }),
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(vec![
                    ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: "Describe this.".to_string(),
                        },
                    ),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl {
                                url: "data:image/png;base64,AAAA".to_string(),
                                detail: Some(ImageDetail::High),
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
                                format: Some("application/pdf".to_string()),
                                detail: None,
                                video_metadata: None,
                            },
                        },
                    ),
                ]),
                name: None,
            }),
        ])
        .model(ModelRef::model("gemini-public"))
        .build()
        .unwrap();

    let response = client.chat().create(&request).await.unwrap();

    assert_eq!(response.choices[0].message.content.as_deref(), Some("ok"));
    let request = last_request(&server).await;
    assert_eq!(
        request
            .headers
            .get("x-goog-api-key")
            .and_then(|value| value.to_str().ok()),
        Some("gemini-test-key")
    );
    let body: Value = request.body_json().unwrap();
    assert_eq!(
        body["systemInstruction"],
        json!({"parts": [{"text": "Be concise."}]})
    );
    assert_eq!(body["contents"][0]["role"], "user");
    assert_eq!(body["contents"][0]["parts"][0]["text"], "Describe this.");
    assert_eq!(
        body["contents"][0]["parts"][1]["inlineData"],
        json!({"mimeType": "image/png", "data": "AAAA"})
    );
    assert_eq!(
        body["generationConfig"]["mediaResolution"],
        "MEDIA_RESOLUTION_HIGH"
    );
    assert!(body["contents"][0]["parts"][1]["mediaResolution"].is_null());
    assert_eq!(
        body["contents"][0]["parts"][2]["inlineData"],
        json!({"mimeType": "audio/wav", "data": "UklGRg=="})
    );
    assert_eq!(
        body["contents"][0]["parts"][3]["inlineData"],
        json!({"mimeType": "application/pdf", "data": "JVBERi0="})
    );
}

#[tokio::test]
async fn gemini_create_maps_openai_params_tools_and_response_format() {
    let server = MockServer::start().await;
    mount_gemini_json(&server, gemini_response("ok")).await;
    let client = Client::build(gemini_config(server.uri())).unwrap();
    let mut request = basic_request();
    request.params.top_p = Some(0.9);
    request.params.stop = Some(StopConfiguration::String("END".to_string()));
    request.params.n = Some(2);
    request.params.reasoning_effort = Some(ReasoningEffort::Low);
    request.params.response_format = Some(ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: Some("Weather response".to_string()),
            name: "weather_response".to_string(),
            schema: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    });
    request.params.tools = Some(vec![ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
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

    client.chat().create(&request).await.unwrap();

    let body = last_body(&server).await;
    assert_eq!(body["generationConfig"]["temperature"], 0.2);
    assert_eq!(body["generationConfig"]["topP"], 0.9);
    assert_eq!(body["generationConfig"]["maxOutputTokens"], 32);
    assert_eq!(body["generationConfig"]["candidateCount"], 2);
    assert_eq!(body["generationConfig"]["stopSequences"], json!(["END"]));
    assert_eq!(
        body["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert_eq!(
        body["generationConfig"]["responseJsonSchema"]["properties"]["city"]["type"],
        "string"
    );
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 1024, "includeThoughts": true})
    );
    assert_eq!(
        body["tools"][0]["functionDeclarations"][0]["name"],
        "get_weather"
    );
    assert_eq!(
        body["toolConfig"]["functionCallingConfig"],
        json!({"mode": "ANY", "allowedFunctionNames": ["get_weather"]})
    );
}

#[tokio::test]
async fn gemini_create_transforms_function_calls_and_usage() {
    let server = MockServer::start().await;
    mount_gemini_json(
        &server,
        json!({
            "responseId": "gemini-tool",
            "candidates": [{
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "Paris"}
                        },
                        "thoughtSignature": "sig-123"
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
                "thoughtsTokenCount": 2,
                "cachedContentTokenCount": 3
            }
        }),
    )
    .await;
    let client = Client::build(gemini_config(server.uri())).unwrap();

    let response = client.chat().create(&basic_request()).await.unwrap();

    assert_eq!(response.id, "gemini-tool");
    assert_eq!(
        response.choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    let tool_call = &response.choices[0].message.tool_calls.as_ref().unwrap()[0];
    let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call else {
        panic!("expected function tool call")
    };
    assert_eq!(tool_call.function.name, "get_weather");
    assert_eq!(tool_call.function.arguments, r#"{"location":"Paris"}"#);
    assert_eq!(
        tool_call.provider_specific_fields.as_ref().unwrap()["thought_signature"],
        "sig-123"
    );
    let usage = response.usage.unwrap();
    assert_eq!(usage.total_tokens, 15);
    assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(3));
    assert_eq!(
        usage.completion_tokens_details.unwrap().reasoning_tokens,
        Some(2)
    );
}

#[tokio::test]
async fn gemini_create_stream_parses_sse_tool_calls_and_usage() {
    let server = MockServer::start().await;
    mount_gemini_stream(
        &server,
        format!(
            "data: {}\n\ndata: {}\n\n",
            json!({
                "responseId": "gemini-stream",
                "candidates": [{
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "Boston"}
                            }
                        }]
                    }
                }]
            }),
            json!({
                "responseId": "gemini-stream",
                "candidates": [{
                    "index": 0,
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 6,
                    "totalTokenCount": 10
                }
            })
        ),
    )
    .await;
    let client = Client::build(gemini_config(server.uri())).unwrap();

    let chunks = client
        .chat()
        .create_stream(&basic_request())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(
        chunks[0].choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    assert_eq!(
        chunks[0].choices[0].delta.tool_calls.as_ref().unwrap()[0]
            .function
            .as_ref()
            .unwrap()
            .name
            .as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        chunks[1].choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    assert_eq!(chunks[1].usage.as_ref().unwrap().total_tokens, 10);
    assert_eq!(last_request(&server).await.url.query(), Some("alt=sse"));
}

#[tokio::test]
async fn gemini_create_maps_error_body_to_provider_business_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
        .respond_with(
            ResponseTemplate::new(StatusCode::BAD_REQUEST.as_u16()).set_body_json(json!({
                "error": {
                    "code": 400,
                    "message": "bad request",
                    "status": "INVALID_ARGUMENT"
                }
            })),
        )
        .mount(&server)
        .await;
    let client = Client::build(gemini_config(server.uri())).unwrap();

    let err = client.chat().create(&basic_request()).await.unwrap_err();

    assert!(matches!(
        err,
        SigmaError::ProviderBusiness {
            provider,
            status,
            code: Some(code),
            message,
            ..
        } if provider == "gemini-primary"
            && status == StatusCode::BAD_REQUEST
            && code == "INVALID_ARGUMENT"
            && message == "bad request"
    ));
}
