use sigma::types::chat::{
    CacheControl, CacheControlTtl, ChatCompletionRequestMessageContentPartAudioArgs,
    ChatCompletionRequestMessageContentPartFile, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartRefusal, ChatCompletionRequestMessageContentPartText,
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
fn text_part_serializes_cache_control_when_set() {
    let p = ChatCompletionRequestMessageContentPartText {
        text: "hello".into(),
        cache_control: Some(CacheControl::ephemeral()),
    };

    let s = serde_json::to_string(&p).unwrap();

    assert_eq!(
        s,
        r#"{"text":"hello","cache_control":{"type":"ephemeral"}}"#
    );
}

#[test]
fn cache_control_serializes_ttl_when_set() {
    let cache_control = CacheControl::ephemeral_with_ttl(CacheControlTtl::OneHour);

    let s = serde_json::to_string(&cache_control).unwrap();

    assert_eq!(s, r#"{"type":"ephemeral","ttl":"1h"}"#);
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
fn image_part_serializes_cache_control_when_set() {
    let p = ChatCompletionRequestMessageContentPartImage {
        image_url: ImageUrl {
            url: "https://x.test/image.png".into(),
            detail: None,
        },
        cache_control: Some(CacheControl::ephemeral()),
    };

    let s = serde_json::to_string(&p).unwrap();

    assert_eq!(
        s,
        r#"{"image_url":{"url":"https://x.test/image.png","detail":null},"cache_control":{"type":"ephemeral"}}"#
    );
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
        cache_control: None,
    };
    let s = serde_json::to_string(&p).unwrap();
    let back: ChatCompletionRequestMessageContentPartFile = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}

#[test]
fn file_part_serializes_cache_control_when_set() {
    let p = ChatCompletionRequestMessageContentPartFile {
        file: FileObject {
            file_id: Some("file_123".into()),
            file_data: None,
            filename: None,
            format: None,
            detail: None,
            video_metadata: None,
        },
        cache_control: Some(CacheControl::ephemeral()),
    };

    let s = serde_json::to_string(&p).unwrap();

    assert_eq!(
        s,
        r#"{"file":{"file_id":"file_123"},"cache_control":{"type":"ephemeral"}}"#
    );
}
