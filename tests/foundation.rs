use sigma::error::SigmaError;

#[test]
fn invalid_argument_displays() {
    let err = SigmaError::InvalidArgument("bad".into());
    assert_eq!(format!("{err}"), "invalid args: bad");
}
