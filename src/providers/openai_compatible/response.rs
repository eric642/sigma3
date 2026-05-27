use serde_json::{Map, Value, json};

pub(super) fn sanitize_null_usage_tokens(value: &mut Value) {
    let Some(usage) = value.get_mut("usage").and_then(Value::as_object_mut) else {
        return;
    };

    for (key, value) in usage {
        if key.ends_with("_tokens") && value.is_null() {
            *value = Value::from(0);
        }
    }
}

pub(super) fn map_response_reasoning_content(value: &mut Value) {
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };

    for choice in choices {
        if let Some(message) = choice.get_mut("message").and_then(Value::as_object_mut) {
            move_reasoning_content(message);
        }
    }
}

pub(super) fn map_stream_reasoning_content(value: &mut Value) {
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };

    for choice in choices {
        if let Some(delta) = choice.get_mut("delta").and_then(Value::as_object_mut) {
            move_reasoning_content(delta);
        }
    }
}

fn move_reasoning_content(object: &mut Map<String, Value>) {
    let Some(reasoning_content) = object.remove("reasoning_content") else {
        return;
    };
    let Some(reasoning_text) = reasoning_content.as_str().filter(|value| !value.is_empty()) else {
        return;
    };
    let reasoning_block = json!({
        "type": "text",
        "text": reasoning_text,
    });

    match object.get_mut("reasoning").and_then(Value::as_array_mut) {
        Some(reasoning) => reasoning.push(reasoning_block),
        None => {
            object.insert("reasoning".to_string(), Value::Array(vec![reasoning_block]));
        }
    }
}
