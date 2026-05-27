//! Shared chat-parameter pipeline used by every provider adapter.
//!
//! The standard chat lifecycle merges deployment defaults with the typed
//! [`ChatRequestParams`](crate::types::chat::ChatRequestParams), then resolves a
//! [`ResolvedChatParamRules`] from the provider's built-in support set and the
//! caller-configured [`ChatParamConfig`], and finally walks the parameter map
//! through four ordered steps:
//!
//! 1. Drop top-level keys listed in [`ChatParamConfig::drop`].
//! 2. Reject or drop keys outside the resolved support set, controlled by the
//!    resolved [`ParamPolicy`].
//! 3. Apply rename rules.
//! 4. Drop nested paths listed in [`ChatParamConfig::drop`] (after renames).
//!
//! Adapters call [`merge_chat_params`] and the rule pipeline from inside
//! `transform_request` so they own the full
//! `ChatRequestParams → native body` translation. Client-layer code only
//! routes and dispatches.

use std::collections::{BTreeMap, HashSet};

use serde_json::Value;

use crate::config::{ChatParamConfig, ChatParamModelConfig, ChatParameterMap, ParamPolicy};
use crate::types::chat::ChatRequest;
use crate::{ModelName, ProviderId, SigmaError, SigmaResult};

/// Resolved chat parameter handling rules for one adapter call.
///
/// Built by [`resolve_chat_param_rules`] from the adapter's built-in support
/// set plus the caller-configured [`ChatParamConfig`] (and an optional
/// model-specific override). Pass to [`apply_chat_param_rules`] alongside the
/// merged parameter map.
///
/// Provider crates implementing [`crate::ChatCompletionAdapter`] use this
/// struct opaquely; only the resolve/apply pair is part of the contract.
pub struct ResolvedChatParamRules {
    policy: ParamPolicy,
    supported: HashSet<String>,
    drop: Vec<String>,
    rename: BTreeMap<String, String>,
}

impl ResolvedChatParamRules {
    fn new(supported: &[&'static str]) -> Self {
        Self {
            policy: ParamPolicy::RejectUnsupported,
            supported: supported.iter().map(|name| (*name).to_string()).collect(),
            drop: Vec::new(),
            rename: BTreeMap::new(),
        }
    }

    fn apply_provider_config(&mut self, config: &ChatParamConfig) {
        if let Some(policy) = config.policy {
            self.policy = policy;
        }
        if let Some(supported) = &config.supported {
            self.supported = supported.iter().cloned().collect();
        }
        self.supported.extend(config.allow.iter().cloned());
        self.drop.extend(config.drop.iter().cloned());
        if let Some(rename) = &config.rename {
            self.rename = rename.clone();
        }
    }

    fn apply_model_config(&mut self, config: &ChatParamModelConfig) {
        if let Some(policy) = config.policy {
            self.policy = policy;
        }
        if let Some(supported) = &config.supported {
            self.supported = supported.iter().cloned().collect();
        }
        self.supported.extend(config.allow.iter().cloned());
        self.drop.extend(config.drop.iter().cloned());
        if let Some(rename) = &config.rename {
            self.rename = rename.clone();
        }
    }

    fn supports(&self, param: &str) -> bool {
        self.supported.contains(param) || self.rename.contains_key(param)
    }
}

/// Merges deployment defaults with the request's typed chat parameters.
///
/// `inject_stream` adds `"stream": true` to the merged map before any rule
/// application so the streaming flag participates in unsupported-parameter
/// validation.
pub fn merge_chat_params(
    deployment_defaults: Option<&ChatParameterMap>,
    request: &ChatRequest,
    inject_stream: bool,
) -> SigmaResult<ChatParameterMap> {
    let mut params = deployment_defaults.cloned().unwrap_or_default();
    params.extend(request.chat_parameters()?);
    if inject_stream {
        params.insert("stream".to_string(), Value::Bool(true));
    }
    Ok(params)
}

/// Resolves provider- and model-level chat parameter rules.
///
/// `default_supported` is the adapter's built-in support set. Optional
/// `chat_param_config` and a matching `provider_model` model entry layer on
/// top, replacing or extending the resolved values.
pub fn resolve_chat_param_rules(
    default_supported: &[&'static str],
    chat_param_config: Option<&ChatParamConfig>,
    provider_model: &ModelName,
) -> ResolvedChatParamRules {
    let mut rules = ResolvedChatParamRules::new(default_supported);
    if let Some(config) = chat_param_config {
        rules.apply_provider_config(config);
        if let Some(model_config) = config.models.get(provider_model) {
            rules.apply_model_config(model_config);
        }
    }
    rules
}

/// Walks the merged parameter map through the four ordered rule steps.
///
/// Returns [`SigmaError::UnsupportedParams`] when the resolved policy is
/// [`ParamPolicy::RejectUnsupported`] and the request contains keys outside
/// the support set. Otherwise, all mutations land on `params` in-place.
pub fn apply_chat_param_rules(
    provider: &ProviderId,
    params: &mut ChatParameterMap,
    rules: &ResolvedChatParamRules,
) -> SigmaResult<()> {
    apply_top_level_drops(params, &rules.drop);

    let unsupported = params
        .keys()
        .filter(|param| !rules.supports(param))
        .cloned()
        .collect::<Vec<_>>();

    if !unsupported.is_empty() {
        match rules.policy {
            ParamPolicy::RejectUnsupported => {
                return Err(SigmaError::UnsupportedParams {
                    provider: provider.clone(),
                    params: unsupported,
                });
            }
            ParamPolicy::DropUnsupported => {
                for param in unsupported {
                    params.remove(&param);
                }
            }
        }
    }

    apply_renames(params, &rules.rename);
    apply_nested_drops(params, &rules.drop);

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParamPathSegment {
    Key(String),
    Wildcard,
    Index(usize),
}

fn apply_top_level_drops(params: &mut ChatParameterMap, drops: &[String]) {
    for drop in drops {
        if !is_nested_param_path(drop) {
            params.remove(drop);
        }
    }
}

fn apply_nested_drops(params: &mut ChatParameterMap, drops: &[String]) {
    if !drops.iter().any(|drop| is_nested_param_path(drop)) {
        return;
    }

    let mut value = Value::Object(std::mem::take(params));
    for drop in drops {
        if is_nested_param_path(drop) {
            delete_nested_param_path(&mut value, &parse_param_path(drop));
        }
    }

    let Value::Object(map) = value else {
        unreachable!("chat parameters are always represented as a JSON object");
    };
    *params = map;
}

fn apply_renames(params: &mut ChatParameterMap, renames: &BTreeMap<String, String>) {
    for (source, target) in renames {
        if let Some(value) = params.remove(source) {
            params.insert(target.clone(), value);
        }
    }
}

fn is_nested_param_path(path: &str) -> bool {
    path.contains('.') || path.contains('[')
}

fn parse_param_path(path: &str) -> Vec<ParamPathSegment> {
    let mut segments = Vec::new();

    for part in path.split('.') {
        let mut rest = part;
        while let Some(index) = rest.find('[') {
            let key = &rest[..index];
            if !key.is_empty() {
                segments.push(ParamPathSegment::Key(key.to_string()));
            }

            let Some(close) = rest[index..].find(']') else {
                return segments;
            };
            let bracket = &rest[index + 1..index + close];
            if bracket == "*" {
                segments.push(ParamPathSegment::Wildcard);
            } else if let Ok(index) = bracket.parse::<usize>() {
                segments.push(ParamPathSegment::Index(index));
            }
            rest = &rest[index + close + 1..];
        }

        if !rest.is_empty() {
            segments.push(ParamPathSegment::Key(rest.to_string()));
        }
    }

    segments
}

fn delete_nested_param_path(value: &mut Value, segments: &[ParamPathSegment]) {
    let Some((segment, rest)) = segments.split_first() else {
        return;
    };

    match segment {
        ParamPathSegment::Key(key) => {
            let Some(object) = value.as_object_mut() else {
                return;
            };
            if rest.is_empty() {
                object.remove(key);
            } else if let Some(value) = object.get_mut(key) {
                delete_nested_param_path(value, rest);
            }
        }
        ParamPathSegment::Wildcard => {
            let Some(array) = value.as_array_mut() else {
                return;
            };
            if !rest.is_empty() {
                for value in array {
                    delete_nested_param_path(value, rest);
                }
            }
        }
        ParamPathSegment::Index(index) => {
            let Some(array) = value.as_array_mut() else {
                return;
            };
            if !rest.is_empty()
                && let Some(value) = array.get_mut(*index)
            {
                delete_nested_param_path(value, rest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rules_with(
        supported: &[&'static str],
        policy: ParamPolicy,
        drop: &[&str],
        rename: &[(&str, &str)],
    ) -> ResolvedChatParamRules {
        let mut rules = ResolvedChatParamRules::new(supported);
        rules.policy = policy;
        rules.drop = drop.iter().map(|item| (*item).to_string()).collect();
        rules.rename = rename
            .iter()
            .map(|(from, to)| ((*from).to_string(), (*to).to_string()))
            .collect();
        rules
    }

    fn params(map: serde_json::Value) -> ChatParameterMap {
        match map {
            Value::Object(map) => map,
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn applies_top_level_drop_before_unsupported_check() {
        let provider = ProviderId::from("p");
        let mut p = params(json!({"logit_bias": 1, "temperature": 0.5}));
        let rules = rules_with(
            &["temperature"],
            ParamPolicy::RejectUnsupported,
            &["logit_bias"],
            &[],
        );

        apply_chat_param_rules(&provider, &mut p, &rules).unwrap();

        assert!(!p.contains_key("logit_bias"));
        assert_eq!(p.get("temperature"), Some(&json!(0.5)));
    }

    #[test]
    fn rejects_unsupported_under_default_policy() {
        let provider = ProviderId::from("p");
        let mut p = params(json!({"weird": 1}));
        let rules = rules_with(&[], ParamPolicy::RejectUnsupported, &[], &[]);

        let err = apply_chat_param_rules(&provider, &mut p, &rules).unwrap_err();
        match err {
            SigmaError::UnsupportedParams { params, .. } => assert_eq!(params, vec!["weird"]),
            other => panic!("expected UnsupportedParams, got {other:?}"),
        }
    }

    #[test]
    fn drops_unsupported_under_drop_policy() {
        let provider = ProviderId::from("p");
        let mut p = params(json!({"weird": 1, "temperature": 0.5}));
        let rules = rules_with(&["temperature"], ParamPolicy::DropUnsupported, &[], &[]);

        apply_chat_param_rules(&provider, &mut p, &rules).unwrap();
        assert!(!p.contains_key("weird"));
        assert_eq!(p.get("temperature"), Some(&json!(0.5)));
    }

    #[test]
    fn renames_then_drops_nested_paths() {
        let provider = ProviderId::from("p");
        let mut p = params(json!({
            "tools": [{"function": {"parameters": {"examples": [1, 2], "type": "object"}}}],
            "max_tokens": 100
        }));
        let rules = rules_with(
            &["tools", "max_completion_tokens"],
            ParamPolicy::RejectUnsupported,
            &["tools[*].function.parameters.examples"],
            &[("max_tokens", "max_completion_tokens")],
        );

        apply_chat_param_rules(&provider, &mut p, &rules).unwrap();

        assert!(p.contains_key("max_completion_tokens"));
        assert!(!p.contains_key("max_tokens"));
        let parameters = &p["tools"][0]["function"]["parameters"];
        assert!(parameters.get("examples").is_none());
        assert_eq!(parameters["type"], json!("object"));
    }

    #[test]
    fn rename_target_keeps_unsupported_source_alive() {
        let provider = ProviderId::from("p");
        let mut p = params(json!({"max_tokens": 100}));
        let rules = rules_with(
            &["max_completion_tokens"],
            ParamPolicy::RejectUnsupported,
            &[],
            &[("max_tokens", "max_completion_tokens")],
        );

        apply_chat_param_rules(&provider, &mut p, &rules).unwrap();
        assert_eq!(p.get("max_completion_tokens"), Some(&json!(100)));
        assert!(!p.contains_key("max_tokens"));
    }
}
