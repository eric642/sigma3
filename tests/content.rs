use sigma::types::chat::{
    ChatCompletionRequestMessageContentPartAudioArgs, ChatCompletionRequestMessageContentPartFile,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartRefusal,
    ChatCompletionRequestMessageContentPartTextArgs, FileObject, InputAudio, InputAudioFormat,
};
use sigma::types::shared::{ImageDetail, ImageUrl};

#[test]
fn text_part_builder() {
    let p = ChatCompletionRequestMessageContentPartTextArgs::default()
        .text("hello")
        .build()
        .unwrap();
    assert_eq!(p.text, "hello");
}

#[test]
fn refusal_part_round_trip() {
    let p = ChatCompletionRequestMessageContentPartRefusal {
        refusal: "no".into(),
    };
    let s = serde_json::to_string(&p).unwrap();
    assert_eq!(s, r#"{"refusal":"no"}"#);
}

#[test]
fn image_part_round_trip() {
    let p = ChatCompletionRequestMessageContentPartImageArgs::default()
        .image_url(ImageUrl {
            url: "https://x.test".into(),
            detail: Some(ImageDetail::High),
        })
        .build()
        .unwrap();
    let s = serde_json::to_string(&p).unwrap();
    let back: ChatCompletionRequestMessageContentPartImage = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}

#[test]
fn input_audio_default_is_mp3() {
    assert_eq!(InputAudioFormat::default(), InputAudioFormat::Mp3);
}

#[test]
fn audio_part_builder() {
    let p = ChatCompletionRequestMessageContentPartAudioArgs::default()
        .input_audio(InputAudio {
            data: "AAAA".into(),
            format: InputAudioFormat::Wav,
        })
        .build()
        .unwrap();
    assert_eq!(p.input_audio.format, InputAudioFormat::Wav);
}

#[test]
fn file_object_skips_none() {
    let f = FileObject {
        file_id: Some("file_123".into()),
        file_data: None,
        filename: None,
        format: None,
        detail: None,
        video_metadata: None,
    };
    let s = serde_json::to_string(&f).unwrap();
    assert_eq!(s, r#"{"file_id":"file_123"}"#);
}

#[test]
fn file_part_round_trip() {
    let p = ChatCompletionRequestMessageContentPartFile {
        file: FileObject {
            file_id: Some("file_123".into()),
            file_data: None,
            filename: None,
            format: None,
            detail: None,
            video_metadata: None,
        },
    };
    let s = serde_json::to_string(&p).unwrap();
    let back: ChatCompletionRequestMessageContentPartFile = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}
