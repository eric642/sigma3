//! Per-model capability metadata used by provider adapters to make routing
//! and translation decisions without resorting to ad-hoc model-name substring
//! checks.
//!
//! sigma ships a vendored snapshot of [models.dev](https://github.com/anomalyco/models.dev)
//! at compile time. Adapters call [`resolve_capabilities`] to look up a model
//! by name (with optional caller-side overrides through
//! [`crate::ModelDeploymentConfig::model_info`]). The lookup falls back through
//! exact match, vendor-family pattern, and finally a portable default so
//! provider code never has to special-case unknown models.
//!
//! Customizing capabilities for a deployment:
//!
//! ```rust
//! use sigma::model_capabilities::{ModelCapabilities, ThinkingMode, VendorFamily, resolve_capabilities};
//! use sigma::ModelName;
//! use serde_json::json;
//!
//! let override_info = json!({
//!     "supports_structured_output": true,
//!     "thinking": "adaptive",
//! });
//! let caps = resolve_capabilities("anthropic", &ModelName::from("custom-model"), Some(&override_info));
//! assert!(caps.supports_structured_output);
//! assert_eq!(caps.thinking, ThinkingMode::Adaptive);
//! let _ = VendorFamily::Other;
//! let _: ModelCapabilities = caps;
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ModelName;

const MODELS_DEV_SNAPSHOT: &str = include_str!("../data/models.dev.json");

/// Boolean and enum flags an adapter consults before mapping a request.
///
/// Fields are intentionally narrow — only what an adapter needs to choose
/// between API paths or refuse a request. Anything richer (token cost, voice
/// support, knowledge cutoff) belongs in
/// [`crate::ModelDeploymentConfig::model_info`] for caller-side use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelCapabilities {
    /// Native structured output is available (Anthropic `output_format`,
    /// OpenAI `response_format` JSON schema, Gemini `responseSchema`).
    pub supports_structured_output: bool,
    /// Provider-specific JSON Schema mode (Gemini `responseJsonSchema`,
    /// OpenAI strict mode). Implies `supports_structured_output`.
    pub supports_response_json_schema: bool,
    /// `frequency_penalty` is valid for this model.
    pub supports_frequency_penalty: bool,
    /// `presence_penalty` is valid for this model.
    pub supports_presence_penalty: bool,
    /// Multiple parallel tool calls are accepted in one assistant turn.
    pub supports_parallel_tool_calls: bool,
    /// Reasoning/thinking shape exposed by the provider.
    pub thinking: ThinkingMode,
    /// Coarse vendor classification adapters use to branch on model family.
    pub vendor_family: VendorFamily,
    /// Provider-reported maximum output tokens, when known.
    pub max_output_tokens: Option<u32>,
    /// Provider-reported context window in tokens, when known.
    pub context_window: Option<u32>,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            supports_structured_output: false,
            supports_response_json_schema: false,
            supports_frequency_penalty: false,
            supports_presence_penalty: false,
            supports_parallel_tool_calls: false,
            thinking: ThinkingMode::None,
            vendor_family: VendorFamily::Other,
            max_output_tokens: None,
            context_window: None,
        }
    }
}

/// Reasoning shape an adapter should send to the provider.
///
/// `None` skips reasoning. `Budget` maps `reasoning_effort` to a
/// `budget_tokens` value (used by Claude 3.5/3.7 and Bedrock Claude). `Adaptive`
/// uses Anthropic's `type = "adaptive"` which lets the model pick its own
/// budget (used by Opus 4.6/4.7 and Sonnet 4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    /// No native reasoning channel.
    None,
    /// Reasoning is enabled with a numeric token budget.
    Budget,
    /// Reasoning is enabled with adaptive (provider-chosen) budgeting.
    Adaptive,
}

/// Coarse model family that adapters branch on.
///
/// Adapters use these instead of substring matching against the model name so
/// renaming a model release does not silently change behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VendorFamily {
    /// Anthropic Claude (any provider; native or Bedrock).
    Claude,
    /// Amazon Nova v1.
    Nova,
    /// Amazon Nova v2.
    Nova2,
    /// OpenAI GPT (Chat Completions / Responses) including o-series.
    Gpt,
    /// OpenAI's open-weight gpt-oss family hosted on Bedrock.
    GptOss,
    /// Google Gemini 1.x/2.x.
    GeminiV2,
    /// Google Gemini 3.x and newer.
    GeminiV3,
    /// Anything else.
    Other,
}

/// Resolves capabilities for a model name under a given provider kind.
///
/// Lookup order:
///
/// 1. `override_info` (deployment-supplied). Missing fields fall through.
/// 2. Exact match on the vendored model snapshot using `provider_kind` plus
///    the model name.
/// 3. Vendor-family pattern derived from the model name.
/// 4. [`ModelCapabilities::default`].
pub fn resolve_capabilities(
    provider_kind: &str,
    model: &ModelName,
    override_info: Option<&Value>,
) -> ModelCapabilities {
    let base =
        lookup_snapshot(provider_kind, model.as_str()).unwrap_or_else(|| ModelCapabilities {
            vendor_family: vendor_family_from_name(provider_kind, model.as_str()),
            ..ModelCapabilities::default()
        });

    apply_override(base, override_info)
}

fn apply_override(mut base: ModelCapabilities, info: Option<&Value>) -> ModelCapabilities {
    let Some(info) = info.and_then(|value| value.as_object()) else {
        return base;
    };

    if let Some(value) = info
        .get("supports_structured_output")
        .and_then(Value::as_bool)
    {
        base.supports_structured_output = value;
    }
    if let Some(value) = info
        .get("supports_response_json_schema")
        .and_then(Value::as_bool)
    {
        base.supports_response_json_schema = value;
    }
    if let Some(value) = info
        .get("supports_frequency_penalty")
        .and_then(Value::as_bool)
    {
        base.supports_frequency_penalty = value;
    }
    if let Some(value) = info
        .get("supports_presence_penalty")
        .and_then(Value::as_bool)
    {
        base.supports_presence_penalty = value;
    }
    if let Some(value) = info
        .get("supports_parallel_tool_calls")
        .and_then(Value::as_bool)
    {
        base.supports_parallel_tool_calls = value;
    }
    if let Some(value) = info.get("thinking").and_then(Value::as_str)
        && let Ok(mode) = serde_json::from_value(Value::String(value.to_string()))
    {
        base.thinking = mode;
    }
    if let Some(value) = info.get("vendor_family").and_then(Value::as_str)
        && let Ok(family) = serde_json::from_value(Value::String(value.to_string()))
    {
        base.vendor_family = family;
    }
    if let Some(value) = info
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
    {
        base.max_output_tokens = Some(value);
    }
    if let Some(value) = info
        .get("context_window")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
    {
        base.context_window = Some(value);
    }

    base
}

fn lookup_snapshot(provider_kind: &str, model: &str) -> Option<ModelCapabilities> {
    let registry = registry();
    let lower = model.to_ascii_lowercase();

    // Prefer the provider-prefixed canonical id (e.g. `anthropic/claude-...`),
    // then plain id, then a stripped suffix (sigma users sometimes pass
    // versioned native names like `claude-3-5-sonnet-20241022`).
    let provider_prefix = provider_to_models_dev_prefix(provider_kind);
    let candidates = [
        provider_prefix
            .map(|prefix| format!("{prefix}/{lower}"))
            .as_deref()
            .unwrap_or("")
            .to_string(),
        lower.clone(),
    ];

    for candidate in &candidates {
        if candidate.is_empty() {
            continue;
        }
        if let Some(caps) = registry.get(candidate) {
            return Some(caps.clone());
        }
    }

    // Substring fallback: vendored ids include version dates whereas callers
    // often configure provider_model without one. Pick the first id that
    // contains the lowercased model name and lives under the right vendor.
    let provider_prefix = provider_prefix.unwrap_or("");
    registry
        .iter()
        .find(|(id, _)| id.starts_with(provider_prefix) && id.contains(&lower))
        .map(|(_, caps)| caps.clone())
}

fn registry() -> &'static HashMap<String, ModelCapabilities> {
    static REGISTRY: OnceLock<HashMap<String, ModelCapabilities>> = OnceLock::new();
    REGISTRY.get_or_init(build_registry)
}

fn build_registry() -> HashMap<String, ModelCapabilities> {
    let snapshot: SnapshotRoot = serde_json::from_str(MODELS_DEV_SNAPSHOT)
        .expect("vendored models.dev snapshot must be valid JSON");
    snapshot
        .data
        .into_iter()
        .map(|entry| (entry.id.to_ascii_lowercase(), capabilities_from(&entry)))
        .collect()
}

fn capabilities_from(entry: &SnapshotEntry) -> ModelCapabilities {
    let supports = |name: &str| {
        entry
            .supported_parameters
            .iter()
            .any(|value| value.eq_ignore_ascii_case(name))
    };
    let supports_structured = supports("structured_outputs") || supports("response_format");
    let supports_reasoning = supports("reasoning") || supports("include_reasoning");
    let thinking = if supports_reasoning {
        if entry.id.contains("opus-4.7")
            || entry.id.contains("opus-4.6")
            || entry.id.contains("sonnet-4.6")
            || entry.id.contains("sonnet-4.7")
        {
            ThinkingMode::Adaptive
        } else {
            ThinkingMode::Budget
        }
    } else {
        ThinkingMode::None
    };
    let max_output_tokens = entry
        .top_provider
        .as_ref()
        .and_then(|provider| provider.max_completion_tokens)
        .and_then(|value| u32::try_from(value).ok());
    let context_window = entry
        .context_length
        .and_then(|value| u32::try_from(value).ok())
        .or_else(|| {
            entry
                .top_provider
                .as_ref()
                .and_then(|provider| provider.context_length)
                .and_then(|value| u32::try_from(value).ok())
        });

    ModelCapabilities {
        supports_structured_output: supports_structured,
        supports_response_json_schema: supports("structured_outputs"),
        supports_frequency_penalty: supports("frequency_penalty"),
        supports_presence_penalty: supports("presence_penalty"),
        // models.dev does not expose a parallel-tool-call flag; assume any
        // model that exposes `tools` accepts the OpenAI parallel mode unless a
        // caller overrides it via deployment.model_info.
        supports_parallel_tool_calls: supports("tools") || supports("tool_choice"),
        thinking,
        vendor_family: vendor_family_from_id(&entry.id),
        max_output_tokens,
        context_window,
    }
}

fn vendor_family_from_id(id: &str) -> VendorFamily {
    let lower = id.to_ascii_lowercase();
    if lower.contains("nova-2") {
        VendorFamily::Nova2
    } else if lower.starts_with("amazon/nova")
        || lower.contains("/nova-")
        || lower.contains("amazon.nova-")
        || lower.contains(".nova-")
    {
        VendorFamily::Nova
    } else if lower.contains("gpt-oss") {
        VendorFamily::GptOss
    } else if lower.starts_with("openai/")
        || lower.contains("gpt-")
        || lower.contains("/o1")
        || lower.contains("/o3")
        || lower.contains("/o4")
    {
        VendorFamily::Gpt
    } else if lower.starts_with("anthropic/") || lower.contains("claude") {
        VendorFamily::Claude
    } else if lower.contains("gemini-3") || lower.contains("gemini-3.") {
        VendorFamily::GeminiV3
    } else if lower.contains("gemini") {
        VendorFamily::GeminiV2
    } else {
        VendorFamily::Other
    }
}

fn vendor_family_from_name(provider_kind: &str, model: &str) -> VendorFamily {
    let synthetic_id = match provider_to_models_dev_prefix(provider_kind) {
        Some(prefix) => format!("{prefix}/{}", model.to_ascii_lowercase()),
        None => model.to_ascii_lowercase(),
    };
    vendor_family_from_id(&synthetic_id)
}

fn provider_to_models_dev_prefix(kind: &str) -> Option<&'static str> {
    match kind {
        "openai" | "openai-compatible" => Some("openai"),
        "anthropic" => Some("anthropic"),
        "gemini" => Some("google"),
        "bedrock" => None, // Bedrock model ids carry their own vendor prefix.
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
struct SnapshotRoot {
    data: Vec<SnapshotEntry>,
}

#[derive(Debug, Deserialize)]
struct SnapshotEntry {
    id: String,
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    top_provider: Option<SnapshotTopProvider>,
    #[serde(default)]
    supported_parameters: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SnapshotTopProvider {
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    max_completion_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn override_takes_precedence_over_snapshot() {
        let override_info = json!({
            "thinking": "adaptive",
            "supports_structured_output": true,
            "max_output_tokens": 4096,
        });
        let caps = resolve_capabilities(
            "anthropic",
            &ModelName::from("claude-3-5-sonnet-20241022"),
            Some(&override_info),
        );
        assert_eq!(caps.thinking, ThinkingMode::Adaptive);
        assert!(caps.supports_structured_output);
        assert_eq!(caps.max_output_tokens, Some(4096));
        assert_eq!(caps.vendor_family, VendorFamily::Claude);
    }

    #[test]
    fn snapshot_lookup_is_case_insensitive() {
        let with_caps =
            resolve_capabilities("anthropic", &ModelName::from("claude-opus-4.7-fast"), None);
        assert_eq!(with_caps.vendor_family, VendorFamily::Claude);
    }

    #[test]
    fn unknown_model_falls_back_to_pattern_default() {
        let caps = resolve_capabilities(
            "unknown-vendor",
            &ModelName::from("totally-made-up-model"),
            None,
        );
        assert_eq!(caps.vendor_family, VendorFamily::Other);
        assert!(!caps.supports_structured_output);
    }

    #[test]
    fn vendor_family_pattern_handles_gpt_oss_on_bedrock() {
        let caps =
            resolve_capabilities("bedrock", &ModelName::from("openai.gpt-oss-120b-1:0"), None);
        assert_eq!(caps.vendor_family, VendorFamily::GptOss);
    }
}
