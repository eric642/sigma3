use http::HeaderMap;
use serde_json::{Map, Number, Value, json};

use crate::ModelName;
use crate::types::chat::{
    ChatMessage, FinishReason, StopConfiguration, UserContent, UserContentPart, VideoMetadata,
};
use crate::types::shared::ImageDetail;

pub(super) fn gemini_model_url(
    api_base: &str,
    version: &str,
    model: &ModelName,
    endpoint: &str,
) -> String {
    let base = api_base.trim_end_matches('/');
    let versioned_base =
        if base.ends_with("/v1") || base.ends_with("/v1beta") || base.ends_with("/v1alpha") {
            base.to_string()
        } else {
            format!("{base}/{version}")
        };
    let model = if model.as_str().starts_with("models/") {
        model.to_string()
    } else {
        format!("models/{model}")
    };
    format!("{versioned_base}/{model}:{endpoint}")
}

pub(super) fn insert_clone(map: &mut Map<String, Value>, key: &str, value: &Value) {
    map.insert(key.to_string(), value.clone());
}

pub(super) fn insert_float(map: &mut Map<String, Value>, key: &str, value: &Value) {
    let Some(number) = value.as_f64() else {
        insert_clone(map, key, value);
        return;
    };
    let rounded = (number * 1_000_000.0).round() / 1_000_000.0;
    let Some(number) = Number::from_f64(rounded) else {
        insert_clone(map, key, value);
        return;
    };
    map.insert(key.to_string(), Value::Number(number));
}

pub(super) fn stop_sequences(value: &Value) -> Value {
    match serde_json::from_value::<StopConfiguration>(value.clone()) {
        Ok(StopConfiguration::String(value)) => json!([value]),
        Ok(StopConfiguration::StringArray(values)) => json!(values),
        Err(_) => Value::Null,
    }
}

pub(super) fn modalities(value: &Value) -> Value {
    let Some(values) = value.as_array() else {
        return Value::Null;
    };
    Value::Array(
        values
            .iter()
            .filter_map(Value::as_str)
            .map(|value| match value {
                "text" => "TEXT",
                "audio" => "AUDIO",
                "image" => "IMAGE",
                _ => "MODALITY_UNSPECIFIED",
            })
            .map(|value| Value::String(value.to_string()))
            .collect(),
    )
}

pub(super) fn gemini_service_tier(value: &str) -> &str {
    match value {
        "auto" => "priority",
        "default" => "standard",
        other => other,
    }
}

pub(super) fn gemini_service_tier_from_headers(
    headers: &HeaderMap,
) -> Option<crate::types::chat::ServiceTier> {
    let value = headers
        .get("x-gemini-service-tier")
        .and_then(|value| value.to_str().ok())?;
    let value = if value.eq_ignore_ascii_case("standard") {
        "default"
    } else {
        value
    };
    serde_json::from_value(Value::String(value.to_ascii_lowercase())).ok()
}

pub(super) fn map_gemini_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY"
        | "RECITATION"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

pub(super) fn parse_function_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

pub(super) fn is_gemini_file_uri(value: &str) -> bool {
    value.starts_with("https://generativelanguage.googleapis.com/")
        || value.starts_with("https://www.googleapis.com/")
}

pub(super) fn media_resolution(detail: Option<&ImageDetail>) -> Option<Value> {
    let level = media_resolution_level(detail?)?;
    Some(json!({ "level": level }))
}

pub(super) fn highest_media_resolution_level(messages: &[ChatMessage]) -> Option<&'static str> {
    let mut best = None;

    for message in messages {
        let ChatMessage::User(message) = message else {
            continue;
        };
        let UserContent::Parts(parts) = &message.content else {
            continue;
        };
        for part in parts {
            let detail = match part {
                UserContentPart::Image(image) => image.image.detail.as_ref(),
                UserContentPart::File(file) => file.file.detail.as_ref(),
                UserContentPart::Text(_) | UserContentPart::Audio(_) => None,
            };
            if detail.is_some_and(|detail| {
                media_resolution_level(detail).is_some()
                    && image_detail_priority(detail) > best.map_or(0, image_detail_priority)
            }) {
                best = detail;
            }
        }
    }

    best.and_then(media_resolution_level)
}

pub(super) fn media_resolution_level(detail: &ImageDetail) -> Option<&'static str> {
    let level = match detail {
        ImageDetail::Low => "MEDIA_RESOLUTION_LOW",
        ImageDetail::Medium => "MEDIA_RESOLUTION_MEDIUM",
        ImageDetail::High => "MEDIA_RESOLUTION_HIGH",
        ImageDetail::UltraHigh | ImageDetail::Original => "MEDIA_RESOLUTION_ULTRA_HIGH",
        ImageDetail::Auto => return None,
    };
    Some(level)
}

pub(super) fn image_detail_priority(detail: &ImageDetail) -> u8 {
    match detail {
        ImageDetail::Auto => 0,
        ImageDetail::Low => 1,
        ImageDetail::Medium => 2,
        ImageDetail::High => 3,
        ImageDetail::UltraHigh | ImageDetail::Original => 4,
    }
}

pub(super) fn gemini_video_metadata(value: &VideoMetadata) -> Option<Value> {
    let mut metadata = Map::new();
    if let Some(fps) = value.fps
        && let Some(number) = Number::from_f64(fps as f64)
    {
        metadata.insert("fps".to_string(), Value::Number(number));
    }
    if let Some(start_offset) = &value.start_offset {
        metadata.insert(
            "startOffset".to_string(),
            Value::String(start_offset.clone()),
        );
    }
    if let Some(end_offset) = &value.end_offset {
        metadata.insert("endOffset".to_string(), Value::String(end_offset.clone()));
    }
    (!metadata.is_empty()).then_some(Value::Object(metadata))
}

pub(super) fn remove_schema_key(value: &mut Value, key: &str) {
    match value {
        Value::Object(object) => {
            object.remove(key);
            for value in object.values_mut() {
                remove_schema_key(value, key);
            }
        }
        Value::Array(values) => {
            for value in values {
                remove_schema_key(value, key);
            }
        }
        _ => {}
    }
}

pub(super) fn add_property_ordering(value: &mut Value) {
    let Value::Object(object) = value else {
        return;
    };
    if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
        let ordering = properties
            .keys()
            .cloned()
            .map(Value::String)
            .collect::<Vec<_>>();
        for value in properties.values_mut() {
            add_property_ordering(value);
        }
        object.insert("propertyOrdering".to_string(), Value::Array(ordering));
    }
    if let Some(items) = object.get_mut("items") {
        add_property_ordering(items);
    }
}

pub(super) fn supports_response_json_schema(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("gemini-2.") || is_gemini_3_or_newer(&model)
}

pub(super) fn is_gemini_3_or_newer(model: &str) -> bool {
    model.to_ascii_lowercase().contains("gemini-3")
}

pub(super) fn u32_field(value: &Value, key: &str) -> u32 {
    value.get(key).and_then(u32_value).unwrap_or_default()
}

pub(super) fn u32_value(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|value| u32::try_from(value).ok())
}
