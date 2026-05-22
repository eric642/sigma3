use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

macro_rules! string_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            #[doc = concat!("Returns the raw string value stored by [`", stringify!($name), "`].")]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name)).field(&self.0).finish()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.0 == *other
            }
        }
    };
}

string_id!(
    ProviderId,
    "Stable identifier for a configured provider instance.\n\nThis is the name used by deployments and direct provider-model routing. It is distinct from provider kind: two instances can share the same kind while using different credentials or endpoints."
);
string_id!(
    DeploymentId,
    "Stable identifier for a model deployment.\n\nA deployment maps a public model name to one configured provider instance and one provider-native model name."
);
string_id!(
    ModelName,
    "Model name used in user-facing requests, deployments, or provider-native calls."
);
string_id!(
    ProviderKind,
    "Runtime provider kind loaded from configuration.\n\nThis value is matched against [`ProviderKindStatic`] values submitted through the provider inventory registry."
);

/// Static provider kind used by inventory registrations.
///
/// Provider registrations must be link-time constants, so the registry uses a
/// `&'static str` kind. Runtime configuration uses [`ProviderKind`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProviderKindStatic(&'static str);

impl ProviderKindStatic {
    /// Creates a static provider kind for use with [`crate::submit_provider!`].
    pub const fn new(value: &'static str) -> Self {
        Self(value)
    }

    /// Returns the provider kind string.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for ProviderKindStatic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl From<ProviderKindStatic> for ProviderKind {
    fn from(value: ProviderKindStatic) -> Self {
        Self::from(value.as_str())
    }
}

/// Strongly typed model selector used by chat requests.
///
/// Use [`ModelRef::model`] for the normal public API, [`ModelRef::deployment`]
/// when a caller needs to target a specific deployment, and
/// [`ModelRef::provider_model`] when bypassing deployment routing.
///
/// `ModelRef` serializes as a plain string so existing OpenAI-compatible JSON
/// request bodies remain unchanged.
///
/// ```rust
/// use sigma::{ModelRef, ProviderId};
///
/// let public = ModelRef::model("gpt-4o");
/// let deployment = ModelRef::deployment("gpt-4o-prod");
/// let direct = ModelRef::provider_model(ProviderId::from("primary"), "provider-model");
///
/// assert_eq!(serde_json::to_string(&public).unwrap(), r#""gpt-4o""#);
/// assert_eq!(deployment.to_string(), "gpt-4o-prod");
/// assert_eq!(direct.to_string(), "primary/provider-model");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelRef {
    /// Route by public model name through configured deployments.
    Model(ModelName),
    /// Route directly to a configured deployment.
    Deployment(DeploymentId),
    /// Route directly to a provider instance and provider-native model name.
    ProviderModel {
        /// Provider instance that should handle the request.
        provider: ProviderId,
        /// Provider-native model name sent to the provider adapter.
        model: ModelName,
    },
}

impl ModelRef {
    /// Selects a model by its public model name.
    pub fn model(model: impl Into<ModelName>) -> Self {
        Self::Model(model.into())
    }

    /// Selects a deployment by its configured deployment id.
    pub fn deployment(deployment: impl Into<DeploymentId>) -> Self {
        Self::Deployment(deployment.into())
    }

    /// Selects a provider instance and provider-native model directly.
    pub fn provider_model(provider: impl Into<ProviderId>, model: impl Into<ModelName>) -> Self {
        Self::ProviderModel {
            provider: provider.into(),
            model: model.into(),
        }
    }
}

impl Default for ModelRef {
    fn default() -> Self {
        Self::Model(ModelName::default())
    }
}

impl From<&str> for ModelRef {
    fn from(value: &str) -> Self {
        Self::model(value)
    }
}

impl From<String> for ModelRef {
    fn from(value: String) -> Self {
        Self::model(value)
    }
}

impl From<ModelName> for ModelRef {
    fn from(value: ModelName) -> Self {
        Self::Model(value)
    }
}

impl PartialEq<&str> for ModelRef {
    fn eq(&self, other: &&str) -> bool {
        match self {
            Self::Model(model) => model.as_str() == *other,
            Self::Deployment(deployment) => deployment.as_str() == *other,
            Self::ProviderModel { model, .. } => model.as_str() == *other,
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(model) => model.fmt(f),
            Self::Deployment(deployment) => deployment.fmt(f),
            Self::ProviderModel { provider, model } => {
                write!(f, "{provider}/{model}")
            }
        }
    }
}

impl Serialize for ModelRef {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Model(model) => serializer.serialize_str(model.as_str()),
            Self::Deployment(deployment) => serializer.serialize_str(deployment.as_str()),
            Self::ProviderModel { model, .. } => serializer.serialize_str(model.as_str()),
        }
    }
}

impl<'de> Deserialize<'de> for ModelRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self::model)
    }
}
