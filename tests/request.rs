use serde_json::json;
use sigma::types::chat::{
    CacheControl, CacheControlTtl, ChatRequest, ChatRequestParams, StopConfiguration, UserMessage,
};
use sigma::{ChatParameterMap, ModelRef, ProviderId};

#[test]
fn stop_configuration_untagged() {
    let s = serde_json::to_string(&StopConfiguration::String("\n".into())).unwrap();
    assert_eq!(s, r#""\n""#);

    let s = serde_json::to_string(&StopConfiguration::StringArray(vec!["a".into()])).unwrap();
    assert_eq!(s, r#"["a"]"#);
}

#[test]
fn chat_request_minimal_skips_empty_params() {
    let req = ChatRequest::new(
        ModelRef::model("gpt-4o"),
        vec![UserMessage::text("hi").into()],
    );

    let s = serde_json::to_string(&req).unwrap();

    assert_eq!(
        s,
        r#"{"messages":[{"role":"user","content":{"text":"hi"}}],"model":"gpt-4o"}"#
    );
}

#[test]
fn chat_request_round_trips_nested_params() {
    let json = r#"{"messages":[{"role":"user","content":{"text":"hi"}}],"model":"gpt-4o","params":{"temperature":0.7}}"#;

    let req: ChatRequest = serde_json::from_str(json).unwrap();

    assert_eq!(req.model, ModelRef::model("gpt-4o"));
    assert_eq!(req.params.temperature, Some(0.7));
}

#[test]
fn chat_request_provider_options_round_trip_as_provider_body_overrides() {
    let mut zhipu_overrides = ChatParameterMap::new();
    zhipu_overrides.insert("model".to_string(), json!("glm-4-plus"));
    zhipu_overrides.insert("metadata".to_string(), json!({"trace_id": "trace-123"}));

    let mut req = ChatRequest::new(
        ModelRef::model("gpt-4o"),
        vec![UserMessage::text("hi").into()],
    );
    req.provider_options
        .insert(ProviderId::from("zhipu"), zhipu_overrides.clone());

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

    let back: ChatRequest = serde_json::from_value(value).unwrap();
    assert_eq!(
        back.provider_options.get(&ProviderId::from("zhipu")),
        Some(&zhipu_overrides)
    );
}

#[test]
fn chat_request_params_serialize_as_semantic_params_object() {
    let req = ChatRequest::new(
        ModelRef::model("gpt-4o"),
        vec![UserMessage::text("hi").into()],
    )
    .with_params(ChatRequestParams {
        temperature: Some(0.7),
        count: Some(2),
        ..Default::default()
    });

    let value = serde_json::to_value(&req).unwrap();

    assert_eq!(value["params"]["temperature"], json!(0.7f32));
    assert_eq!(value["params"]["count"], json!(2));
    assert!(value.get("temperature").is_none());
    assert!(value.get("count").is_none());
}

#[test]
fn chat_request_params_serializes_typed_cache_control() {
    let req = ChatRequest::new(
        ModelRef::model("claude-public"),
        vec![UserMessage::text("hi").into()],
    )
    .with_params(ChatRequestParams {
        cache_control: Some(CacheControl::ephemeral_with_ttl(
            CacheControlTtl::FiveMinutes,
        )),
        ..Default::default()
    });

    let value = serde_json::to_value(&req).unwrap();

    assert_eq!(
        value["params"]["cache_control"],
        json!({"type": "ephemeral", "ttl": "5m"})
    );
}
