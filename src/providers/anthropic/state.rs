use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::ChatAdapterContext;

use super::TOOL_NAME_MAP_STATE_KEY;

pub(super) fn reverse_tool_map(context: &ChatAdapterContext<'_>) -> HashMap<String, String> {
    context
        .provider_state
        .as_ref()
        .and_then(|state| state.get(TOOL_NAME_MAP_STATE_KEY))
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn current_unix_timestamp() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u32::try_from(duration.as_secs()).ok())
        .unwrap_or(u32::MAX)
}
