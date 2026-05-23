use sigma::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage, ChatCompletionTool,
    ChatCompletionTools, CreateChatCompletionRequestArgs, CreateChatCompletionRequestParamsArgs,
    CreateChatCompletionResponse,
};
use sigma::types::shared::FunctionObject;

#[test]
fn build_request_with_tool() {
    let req = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage::from("hello"),
        )])
        .model("gpt-4o")
        .params(
            CreateChatCompletionRequestParamsArgs::default()
                .tools(vec![ChatCompletionTools::Function(ChatCompletionTool {
                    function: FunctionObject {
                        name: "get_weather".into(),
                        description: Some("Get the weather".into()),
                        parameters: Some(serde_json::json!({"type":"object","properties":{}})),
                        strict: None,
                    },
                })])
                .temperature(0.7f32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let s = serde_json::to_string(&req).unwrap();
    assert!(s.contains(r#""model":"gpt-4o""#));
    assert!(s.contains(r#""tools":[{"type":"function""#));
}

#[test]
fn parse_real_world_response() {
    let json = r#"{
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }
        ],
        "created": 1700000000,
        "model": "gpt-4o",
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2
        }
    }"#;
    let r: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(r.id, "chatcmpl-1");
    assert_eq!(r.choices[0].message.content.as_deref(), Some("Hello!"));
    assert_eq!(r.usage.unwrap().total_tokens, 2);
}
