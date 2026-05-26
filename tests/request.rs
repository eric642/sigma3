use std::collections::HashMap;

use serde_json::json;
use sigma::types::chat::{
    CacheControl, CacheControlTtl, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, CreateChatCompletionRequestParamsArgs, Prompt,
    StopConfiguration,
};
use sigma::{ChatParameterMap, ProviderId};

#[test]
fn prompt_untagged_string() {
    let s = serde_json::to_string(&Prompt::String("hi".into())).unwrap();
    assert_eq!(s, r#""hi""#);
}

#[test]
fn prompt_untagged_string_array() {
    let s = serde_json::to_string(&Prompt::StringArray(vec!["a".into(), "b".into()])).unwrap();
    assert_eq!(s, r#"["a","b"]"#);
}

#[test]
fn stop_configuration_untagged() {
    let s = serde_json::to_string(&StopConfiguration::String("\n".into())).unwrap();
    assert_eq!(s, r#""\n""#);
    let s = serde_json::to_string(&StopConfiguration::StringArray(vec!["a".into()])).unwrap();
    assert_eq!(s, r#"["a"]"#);
}

#[test]
fn create_request_minimal_skips_none() {
    let user = ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Text("hi".into()),
        name: None,
    };
    let req = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(user)])
        .model("gpt-4o")
        .build()
        .unwrap();
    let s = serde_json::to_string(&req).unwrap();
    assert_eq!(
        s,
        r#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"}"#
    );
}

#[test]
fn create_request_round_trip() {
    let json =
        r#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","temperature":0.7}"#;
    let req: CreateChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "gpt-4o");
    assert_eq!(req.params.temperature, Some(0.7));
}

#[test]
fn create_request_provider_options_round_trips_as_provider_body_overrides() {
    let mut zhipu_overrides = ChatParameterMap::new();
    zhipu_overrides.insert("model".to_string(), json!("glm-4-plus"));
    zhipu_overrides.insert("metadata".to_string(), json!({"trace_id": "trace-123"}));

    let req = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage::from("hi"),
        )])
        .model("gpt-4o")
        .provider_options(HashMap::from([(
            ProviderId::from("zhipu"),
            zhipu_overrides.clone(),
        )]))
        .build()
        .unwrap();

    let value = serde_json::to_value(&req).unwrap();
    assert_eq!(
        value["provider_options"],
        json!({
            "zhipu": {
                "metadata": {
                    "trace_id": "trace-123",
                },
                "model": "glm-4-plus",
            },
        })
    );

    let back: CreateChatCompletionRequest = serde_json::from_value(value).unwrap();
    assert_eq!(
        back.provider_options.get(&ProviderId::from("zhipu")),
        Some(&zhipu_overrides)
    );
}

#[test]
fn create_request_params_flatten_into_request_json() {
    let params = CreateChatCompletionRequestParamsArgs::default()
        .temperature(0.7)
        .n(2)
        .build()
        .unwrap();
    let req = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage::from("hi"),
        )])
        .model("gpt-4o")
        .params(params)
        .build()
        .unwrap();

    let value = serde_json::to_value(&req).unwrap();

    assert_eq!(value["temperature"], json!(0.7f32));
    assert_eq!(value["n"], json!(2));
    assert!(value.get("params").is_none());
}

#[test]
fn create_request_params_serializes_typed_cache_control() {
    let params = CreateChatCompletionRequestParamsArgs::default()
        .cache_control(CacheControl::ephemeral_with_ttl(
            CacheControlTtl::FiveMinutes,
        ))
        .build()
        .unwrap();
    let req = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage::from("hi"),
        )])
        .model("claude-public")
        .params(params)
        .build()
        .unwrap();

    let value = serde_json::to_value(&req).unwrap();

    assert_eq!(
        value["cache_control"],
        json!({"type": "ephemeral", "ttl": "5m"})
    );
}
