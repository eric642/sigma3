use std::collections::{BTreeSet, HashMap};

use serde_json::{Map, Value, json};

use crate::config::ChatParameterMap;
use crate::{ProviderId, SigmaError, SigmaResult};

#[derive(Default)]
pub(super) struct ToolNameMaps {
    pub(super) forward: HashMap<String, String>,
    pub(super) reverse: HashMap<String, String>,
}

impl ToolNameMaps {
    pub(super) fn has_rewrites(&self) -> bool {
        !self.forward.is_empty()
    }
}

pub(super) fn prepare_tools(params: &mut ChatParameterMap) -> SigmaResult<ToolNameMaps> {
    let Some(value) = params.get_mut("tools") else {
        return Ok(ToolNameMaps::default());
    };
    let Some(tools) = value.as_array_mut() else {
        return Ok(ToolNameMaps::default());
    };

    for tool in tools.iter_mut() {
        if tool.get("input_schema").is_some() {
            continue;
        }
        let mapped = map_openai_tool(tool)?;
        *tool = mapped;
    }

    let names = tools
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("custom"))
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let maps = build_tool_name_maps(&names);
    if maps.has_rewrites() {
        for tool in tools {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            if let Some(mapped) = maps.forward.get(name).cloned()
                && let Some(object) = tool.as_object_mut()
            {
                object.insert("name".to_string(), Value::String(mapped));
            }
        }
    }

    Ok(maps)
}

fn map_openai_tool(tool: &Value) -> SigmaResult<Value> {
    let Some(object) = tool.as_object() else {
        return Ok(tool.clone());
    };
    let tool_type = object.get("type").and_then(Value::as_str);
    if tool_type == Some("function") {
        let function = object
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| SigmaError::ProviderTransform {
                provider: ProviderId::from("anthropic"),
                message: "function tool is missing function object".to_string(),
            })?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("tool")
            .to_string();
        let mut mapped = Map::new();
        mapped.insert("type".to_string(), Value::String("custom".to_string()));
        mapped.insert("name".to_string(), Value::String(name));
        if let Some(description) = function.get("description") {
            mapped.insert("description".to_string(), description.clone());
        }

        let mut input_schema = function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        if input_schema.get("type").and_then(Value::as_str) != Some("object")
            && let Some(input_schema) = input_schema.as_object_mut()
        {
            input_schema.insert("type".to_string(), Value::String("object".to_string()));
            input_schema
                .entry("properties")
                .or_insert_with(|| Value::Object(Map::new()));
        }
        mapped.insert("input_schema".to_string(), input_schema);
        Ok(Value::Object(mapped))
    } else {
        Ok(tool.clone())
    }
}

fn build_tool_name_maps(names: &[String]) -> ToolNameMaps {
    let mut forward = HashMap::new();
    let mut used = BTreeSet::new();

    for original in names {
        let candidate = sanitize_tool_name(original);
        if candidate == *original {
            used.insert(candidate);
        }
    }

    for original in names {
        let candidate = sanitize_tool_name(original);
        if candidate == *original || forward.contains_key(original) {
            continue;
        }

        let mut unique = candidate.clone();
        let mut suffix = 1;
        while used.contains(&unique) {
            suffix += 1;
            let suffix_value = format!("_{suffix}");
            let max_head = 128usize.saturating_sub(suffix_value.len());
            unique = format!("{}{}", truncate_chars(&candidate, max_head), suffix_value);
        }
        forward.insert(original.clone(), unique.clone());
        used.insert(unique);
    }

    let reverse = forward
        .iter()
        .map(|(original, mapped)| (mapped.clone(), original.clone()))
        .collect();

    ToolNameMaps { forward, reverse }
}

fn sanitize_tool_name(name: &str) -> String {
    truncate_chars(
        &name
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect::<String>(),
        128,
    )
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

pub(super) fn apply_tool_choice_name_map(
    params: &mut ChatParameterMap,
    forward: &HashMap<String, String>,
) {
    let Some(tool_choice) = params.get_mut("tool_choice") else {
        return;
    };
    if let Some(object) = tool_choice.as_object_mut() {
        if let Some(name) = object.get("name").and_then(Value::as_str) {
            if let Some(mapped) = forward.get(name).cloned() {
                object.insert("name".to_string(), Value::String(mapped));
            }
        } else if let Some(function) = object.get_mut("function").and_then(Value::as_object_mut)
            && let Some(name) = function.get("name").and_then(Value::as_str)
            && let Some(mapped) = forward.get(name).cloned()
        {
            function.insert("name".to_string(), Value::String(mapped));
        }
    }
}
