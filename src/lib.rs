//! sigma - a general-purpose LLM API client.
//!
//! sigma separates the user-facing client API from provider-specific driver
//! code. Applications build a [`Client`] from a [`ClientConfig`], then call the
//! namespaced chat API:
//! `client.chat().create(request).await` or
//! `client.chat().create_stream(request).await`.
//!
//! ```no_run
//! # use std::collections::HashMap;
//! use sigma::{
//!     Client, ClientConfig, ModelDeploymentConfig, ModelName, ParamPolicy,
//!     ProviderId, ProviderInstanceConfig, ProviderKind,
//! };
//!
//! # fn build_client() -> sigma::SigmaResult<Client> {
//! let config = ClientConfig {
//!     providers: vec![ProviderInstanceConfig {
//!         id: ProviderId::from("primary"),
//!         kind: ProviderKind::from("openai"),
//!         api_base: None,
//!         api_key: None,
//!         headers: HashMap::new(),
//!         options: serde_json::Value::Null,
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
//!     param_policy: ParamPolicy::RejectUnsupported,
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

mod client;
mod config;
pub mod error;
mod model;
mod provider;
mod provider_http;
pub mod types;

#[doc(hidden)]
pub use inventory;

pub use client::*;
pub use config::*;
pub use error::{SigmaError, SigmaResult};
pub use model::*;
pub use provider::*;
pub use provider_http::*;
