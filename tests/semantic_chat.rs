use serde_json::json;
use sigma::types::chat::{
    AllowedToolsChoice, CacheControl, CacheControlTtl, ChatMessage, ChatRequest, ChatRequestParams,
    FileInput, FunctionTool, ImagePart, ProviderContextBlock, ReasoningBlock, TextPart, ToolChoice,
    ToolChoiceAllowedMode, ToolDefinition, UserContent, UserContentPart, UserMessage,
    VideoMetadata,
};
use sigma::types::shared::FunctionObject;
use sigma::{ModelRef, ProviderId};

#[test]
fn chat_request_serializes_params_object() {
    let request = ChatRequest::new(
        ModelRef::model("model-public"),
        vec![UserMessage::text("hello").into()],
    )
    .with_params(ChatRequestParams {
        temperature: Some(0.7),
        n: Some(2),
        cache_control: Some(CacheControl::ephemeral_with_ttl(
            CacheControlTtl::FiveMinutes,
        )),
        ..Default::default()
    });

    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value,
        json!({
            "messages": [{"role": "user", "content": {"text": "hello"}}],
            "model": "model-public",
            "params": {
                "n": 2,
                "temperature": 0.7f32,
                "cache_control": {"type": "ephemeral", "ttl": "5m"}
            }
        })
    );
}

#[test]
fn chat_request_provider_options_remain_provider_scoped_escape_hatch() {
    let request = ChatRequest::new(
        ModelRef::model("model-public"),
        vec![UserMessage::text("hello").into()],
    )
    .with_provider_option(
        ProviderId::from("selected"),
        "native_flag",
        json!({"enabled": true}),
    );

    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["provider_options"],
        json!({"selected": {"native_flag": {"enabled": true}}})
    );
}

#[test]
fn reasoning_blocks_round_trip_without_provider_specific_fields() {
    let blocks = vec![
        ReasoningBlock::text("thinking", Some("sig-1")),
        ReasoningBlock::redacted("ciphertext", Some("sig-2")),
        ReasoningBlock::signature("gemini-signature"),
    ];

    let value = serde_json::to_value(&blocks).unwrap();
    let back: Vec<ReasoningBlock> = serde_json::from_value(value).unwrap();

    assert_eq!(back, blocks);
}

#[test]
fn provider_context_blocks_round_trip_as_opaque_replay_context() {
    let block = ProviderContextBlock::new(
        "anthropic-primary",
        "anthropic.content_block",
        json!({"type": "compaction", "content": "summary"}),
    );

    let value = serde_json::to_value(&block).unwrap();
    let back: ProviderContextBlock = serde_json::from_value(value).unwrap();

    assert_eq!(back, block);
}

#[test]
fn video_metadata_is_typed_on_file_input() {
    let file = FileInput {
        data: Some("data:video/mp4;base64,AAAA".to_string()),
        id: None,
        filename: Some("clip.mp4".to_string()),
        media_type: Some("video/mp4".to_string()),
        detail: None,
        video_metadata: Some(VideoMetadata {
            fps: Some(24.0),
            start_offset: Some("0s".to_string()),
            end_offset: Some("3s".to_string()),
        }),
    };

    let value = serde_json::to_value(file).unwrap();

    assert_eq!(value["video_metadata"]["fps"], json!(24.0));
}

#[test]
fn allowed_tools_choice_uses_typed_tool_definitions() {
    let choice = ToolChoice::allowed(AllowedToolsChoice {
        mode: ToolChoiceAllowedMode::Required,
        tools: vec![ToolDefinition::Function(FunctionTool {
            function: FunctionObject {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        })],
    });

    let value = serde_json::to_value(choice).unwrap();

    assert_eq!(value["type"], json!("allowed"));
    assert_eq!(value["tools"][0]["type"], json!("function"));
}

#[test]
fn user_content_parts_are_semantic_not_openai_wire_shape() {
    let message = ChatMessage::User(UserMessage {
        content: UserContent::Parts(vec![
            TextPart::new("describe").into(),
            UserContentPart::Image(ImagePart::from_url("data:image/png;base64,AAAA")),
        ]),
        name: None,
    });

    let value = serde_json::to_value(message).unwrap();

    assert_eq!(value["role"], json!("user"));
    assert!(value["content"]["parts"][0].get("type").is_some());
}
