use sigma::types::chat::{
    AudioPart, CacheControl, CacheControlTtl, FileInput, FilePart, ImagePart, InputAudio,
    InputAudioFormat, RefusalPart, TextPart,
};
use sigma::types::shared::{ImageDetail, ImageUrl};

#[test]
fn text_part_new_sets_text() {
    let p = TextPart::new("hello");

    assert_eq!(p.text, "hello");
}

#[test]
fn text_part_serializes_cache_control_when_set() {
    let p = TextPart {
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
    let p = RefusalPart {
        refusal: "no".into(),
    };

    let s = serde_json::to_string(&p).unwrap();
    let back: RefusalPart = serde_json::from_str(&s).unwrap();

    assert_eq!(p, back);
}

#[test]
fn image_part_round_trip() {
    let p = ImagePart {
        image: ImageUrl {
            url: "https://x.test".into(),
            detail: Some(ImageDetail::High),
        },
        cache_control: None,
    };

    let s = serde_json::to_string(&p).unwrap();
    let back: ImagePart = serde_json::from_str(&s).unwrap();

    assert_eq!(p, back);
}

#[test]
fn image_part_serializes_cache_control_when_set() {
    let p = ImagePart::from_url("https://x.test/image.png")
        .with_cache_control(CacheControl::ephemeral());

    let s = serde_json::to_string(&p).unwrap();

    assert_eq!(
        s,
        r#"{"image":{"url":"https://x.test/image.png","detail":null},"cache_control":{"type":"ephemeral"}}"#
    );
}

#[test]
fn input_audio_default_is_mp3() {
    assert_eq!(InputAudioFormat::default(), InputAudioFormat::Mp3);
}

#[test]
fn audio_part_holds_input_audio() {
    let p = AudioPart {
        input_audio: InputAudio {
            data: "AAAA".into(),
            format: InputAudioFormat::Wav,
        },
    };

    assert_eq!(p.input_audio.format, InputAudioFormat::Wav);
}

#[test]
fn file_object_skips_none() {
    let f = FileInput {
        id: Some("file_123".into()),
        data: None,
        filename: None,
        media_type: None,
        detail: None,
        video_metadata: None,
    };

    let s = serde_json::to_string(&f).unwrap();

    assert_eq!(s, r#"{"id":"file_123"}"#);
}

#[test]
fn file_part_round_trip() {
    let p = FilePart {
        file: FileInput {
            id: Some("file_123".into()),
            data: None,
            filename: None,
            media_type: None,
            detail: None,
            video_metadata: None,
        },
        cache_control: None,
    };

    let s = serde_json::to_string(&p).unwrap();
    let back: FilePart = serde_json::from_str(&s).unwrap();

    assert_eq!(p, back);
}

#[test]
fn file_part_serializes_cache_control_when_set() {
    let p = FilePart {
        file: FileInput {
            id: Some("file_123".into()),
            data: None,
            filename: None,
            media_type: None,
            detail: None,
            video_metadata: None,
        },
        cache_control: Some(CacheControl::ephemeral()),
    };

    let s = serde_json::to_string(&p).unwrap();

    assert_eq!(
        s,
        r#"{"file":{"id":"file_123"},"cache_control":{"type":"ephemeral"}}"#
    );
}
