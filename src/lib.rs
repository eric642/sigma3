// SigmaError carries provider-attribution metadata (provider id, status,
// optional structured details) on every variant so callers do not have to do a
// second lookup. The struct is large by design; boxing the inner payload would
// force unwrap-and-rebind everywhere callers match on it without any
// performance benefit on the hot path. Document the trade-off here once.
#![allow(clippy::result_large_err)]

//! sigma - a general-purpose LLM API client.
//!
//! sigma separates the user-facing client API from provider-specific driver
//! code. Applications build a [`Client`] from a [`ClientConfig`], then call the
//! direct chat API:
//! `client.create(&request).await` or
//! `client.create_stream(&request).await`.
//!
//! ```no_run
//! use sigma::{
//!     Client, ClientConfig, ModelDeploymentConfig, ModelName, ProviderCommonConfig,
//!     ProviderConfigMap, ProviderId, ProviderInstanceConfig, ProviderKind,
//! };
//!
//! # fn build_client() -> sigma::SigmaResult<Client> {
//! let config = ClientConfig {
//!     providers: vec![ProviderInstanceConfig {
//!         id: ProviderId::from("primary"),
//!         kind: ProviderKind::from("openai"),
//!         common: ProviderCommonConfig::default(),
//!         config: ProviderConfigMap::new(),
//!     }],
//!     deployments: vec![ModelDeploymentConfig {
//!         id: "gpt-4o-prod".into(),
//!         public_model: ModelName::from("gpt-4o"),
//!         provider: ProviderId::from("primary"),
//!         provider_model: ModelName::from("gpt-4o-2024-08-06"),
//!         defaults: serde_json::Map::new(),
//!         model_info: serde_json::Value::Null,
//!     }],
//!     default_model: Some(ModelName::from("gpt-4o")),
//! };
//!
//! let client = Client::build(config)?;
//! # Ok(client)
//! # }
//! ```
//!
//! Provider drivers are discovered through [`submit_provider!`]. A provider
//! crate registers a static [`ProviderRegistration`], and sigma instantiates
//! provider instances from [`ClientConfig::providers`] at [`Client::build`]
//! time. Model routing is explicit through [`ModelRef`]; sigma does not infer a
//! provider from model-name prefixes.
//!
//! sigma links built-in chat providers for OpenAI (`kind = "openai"`) and
//! OpenAI-compatible HTTP endpoints (`kind = "openai-compatible"`). Simple
//! provider crates that expose the same wire shape can register their own kind
//! and delegate construction to [`OpenAiCompatibleProvider`]. All providers use
//! the standard chat completion methods and can be selected through
//! deployment routing or [`ModelRef::provider_model`].

mod client;
mod config;
pub mod error;
mod model;
pub mod model_capabilities;
mod provider;
mod provider_http;
mod providers;
pub mod types;

#[doc(inline)]
pub use providers::{
    OpenAiCompatibleConfig, OpenAiCompatibleProvider, OpenAiCompatibleProviderSpec,
    ResolvedChatParamRules, apply_chat_param_rules, merge_chat_params, resolve_chat_param_rules,
};

#[doc(hidden)]
pub use inventory;

#[doc(hidden)]
pub mod __private {
    pub use serde_json;
}

pub use client::*;
pub use config::*;
pub use error::{SigmaError, SigmaResult};
pub use model::*;
pub use provider::*;
pub use provider_http::*;
