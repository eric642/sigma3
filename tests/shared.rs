use sigma::types::shared::{
    CompletionTokensDetails, CustomGrammarFormatParam, FunctionCall, FunctionName,
    FunctionObjectArgs, GrammarSyntax, ImageDetail, ImageUrlArgs, PromptTokensDetails,
    ReasoningEffort, ResponseFormat, ResponseFormatJsonSchema,
};

#[test]
fn completion_tokens_details_round_trip() {
    let v = CompletionTokensDetails {
        accepted_prediction_tokens: Some(1),
        audio_tokens: Some(2),
        reasoning_tokens: Some(3),
        rejected_prediction_tokens: Some(4),
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: CompletionTokensDetails = serde_json::from_str(&s).unwrap();
    assert_eq!(v, back);
}

#[test]
fn prompt_tokens_details_round_trip() {
    let v: PromptTokensDetails =
        serde_json::from_str(r#"{"audio_tokens":1,"cached_tokens":2}"#).unwrap();
    assert_eq!(v.audio_tokens, Some(1));
    assert_eq!(v.cached_tokens, Some(2));
}

#[test]
fn function_call_round_trip() {
    let v = FunctionCall {
        name: "f".into(),
        arguments: "{}".into(),
    };
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(s, r#"{"name":"f","arguments":"{}"}"#);
}

#[test]
fn function_name_round_trip() {
    let s = serde_json::to_string(&FunctionName { name: "f".into() }).unwrap();
    assert_eq!(s, r#"{"name":"f"}"#);
}

#[test]
fn function_object_builder() {
    let v = FunctionObjectArgs::default()
        .name("get_weather")
        .description("Get the weather")
        .build()
        .unwrap();
    assert_eq!(v.name, "get_weather");
    assert_eq!(v.description.as_deref(), Some("Get the weather"));
}

#[test]
fn image_detail_default_is_auto() {
    assert_eq!(ImageDetail::default(), ImageDetail::Auto);
    let s = serde_json::to_string(&ImageDetail::High).unwrap();
    assert_eq!(s, r#""high""#);
}

#[test]
fn image_url_builder() {
    let v = ImageUrlArgs::default()
        .url("https://x.test/y.png")
        .build()
        .unwrap();
    assert_eq!(v.url, "https://x.test/y.png");
}

#[test]
fn reasoning_effort_default_is_medium() {
    assert_eq!(ReasoningEffort::default(), ReasoningEffort::Medium);
    let s = serde_json::to_string(&ReasoningEffort::Low).unwrap();
    assert_eq!(s, r#""low""#);
}

#[test]
fn response_format_text() {
    let s = serde_json::to_string(&ResponseFormat::Text).unwrap();
    assert_eq!(s, r#"{"type":"text"}"#);
}

#[test]
fn response_format_json_schema() {
    let v = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: None,
            name: "Foo".into(),
            schema: Some(serde_json::json!({"type":"object"})),
            strict: Some(true),
        },
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: ResponseFormat = serde_json::from_str(&s).unwrap();
    assert_eq!(v, back);
}

#[test]
fn custom_grammar_format_param_round_trip() {
    let v = CustomGrammarFormatParam {
        definition: ".+".into(),
        syntax: GrammarSyntax::Regex,
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: CustomGrammarFormatParam = serde_json::from_str(&s).unwrap();
    assert_eq!(v, back);
}
