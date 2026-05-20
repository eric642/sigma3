use sigma::error::SigmaError;
use sigma::types::Metadata;

#[test]
fn invalid_argument_displays() {
    let err = SigmaError::InvalidArgument("bad".into());
    assert_eq!(format!("{err}"), "invalid args: bad");
}

#[test]
fn metadata_round_trips() {
    let raw = serde_json::json!({"k": "v"});
    let meta: Metadata = raw.clone().into();
    let s = serde_json::to_string(&meta).unwrap();
    assert_eq!(s, r#"{"k":"v"}"#);
    let back: Metadata = serde_json::from_str(&s).unwrap();
    assert_eq!(meta, back);
}
