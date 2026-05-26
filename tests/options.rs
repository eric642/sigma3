use sigma::types::chat::{
    AudioOutput, AudioOutputFormat, AudioVoice, OutputModality, PredictionContent,
    PredictionContentValue, ServiceTier, StreamOptions, Verbosity, WebSearchContextSize,
    WebSearchOptions, WebSearchUserLocationType,
};

#[test]
fn service_tier_lowercase() {
    let s = serde_json::to_string(&ServiceTier::Auto).unwrap();
    assert_eq!(s, r#""auto""#);
}

#[test]
fn verbosity_default_medium() {
    assert_eq!(Verbosity::default(), Verbosity::Medium);
}

#[test]
fn modalities_lowercase() {
    let s = serde_json::to_string(&OutputModality::Audio).unwrap();
    assert_eq!(s, r#""audio""#);
}

#[test]
fn web_search_context_default_medium() {
    assert_eq!(
        WebSearchContextSize::default(),
        WebSearchContextSize::Medium
    );
}

#[test]
fn web_search_options_default() {
    let v = WebSearchOptions::default();
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(s, r#"{}"#);
}

#[test]
fn user_location_type_lowercase() {
    let s = serde_json::to_string(&WebSearchUserLocationType::Approximate).unwrap();
    assert_eq!(s, r#""approximate""#);
}

#[test]
fn prediction_content_string() {
    let v = PredictionContent::Content(PredictionContentValue::Text("hi".into()));
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(s, r#"{"type":"content","content":"hi"}"#);
}

#[test]
fn audio_voice_known_and_other() {
    let s = serde_json::to_string(&AudioVoice::Alloy).unwrap();
    assert_eq!(s, r#""alloy""#);
    let custom = AudioVoice::Other("marin".into());
    let s = serde_json::to_string(&custom).unwrap();
    assert_eq!(s, r#""marin""#);
}

#[test]
fn audio_format_lowercase() {
    let s = serde_json::to_string(&AudioOutputFormat::Pcm16).unwrap();
    assert_eq!(s, r#""pcm16""#);
}

#[test]
fn chat_completion_audio_round_trip() {
    let a = AudioOutput {
        voice: AudioVoice::Alloy,
        format: AudioOutputFormat::Mp3,
    };
    let s = serde_json::to_string(&a).unwrap();
    let back: AudioOutput = serde_json::from_str(&s).unwrap();
    assert_eq!(a, back);
}

#[test]
fn stream_options_round_trip() {
    let o = StreamOptions {
        include_usage: Some(true),
        include_obfuscation: None,
    };
    let s = serde_json::to_string(&o).unwrap();
    assert_eq!(s, r#"{"include_usage":true}"#);
}
