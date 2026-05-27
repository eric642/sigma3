use std::any::Any;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_core::Stream;
use schemars::{JsonSchema, generate::SchemaSettings};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::config::{
    ChatParameterMap, ProviderCommonConfig, ProviderConfigMap, ProviderInstanceConfig,
};
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, ProviderState,
    SignedProviderRequest,
};
use crate::types::chat::{ChatRequest, ChatResponse, ChatStreamChunk};
use crate::{
    DeploymentId, ModelDeploymentConfig, ModelName, ProviderId, ProviderKind, ProviderKindStatic,
    SigmaError, SigmaResult,
};

/// Function pointer used to initialize one configured provider instance.
///
/// Constructors are plain function pointers so provider registrations can be
/// submitted as static inventory data.
pub type ProviderConstructor = fn(ProviderInit) -> SigmaResult<Arc<dyn ProviderDriver>>;

/// Function pointer used by typed provider constructors.
///
/// Provider crates normally do not name this type directly. They pass a
/// function with this shape to [`crate::submit_provider!`], and sigma generates
/// the erased inventory wrapper that deserializes the provider-specific
/// configuration object before calling it.
pub type TypedProviderConstructor<TConfig> =
    fn(ProviderInit<TConfig>) -> SigmaResult<Arc<dyn ProviderDriver>>;

#[doc(hidden)]
pub type ProviderInstanceConfigSchemaFn = fn(ProviderKindStatic) -> Value;

/// Static provider registration collected by the inventory registry.
///
/// Provider crates normally create registrations with [`crate::submit_provider!`]
/// instead of constructing this type directly. The macro expands into a
/// `ProviderRegistration` with the typed `config` deserialization wired through
/// a function pointer, then submits it through `inventory::submit!` so
/// [`ProviderCatalog::from_inventory`] can collect it during
/// [`crate::Client::build`]. Each registration must use a unique
/// [`ProviderKindStatic`]; duplicates are rejected at catalog construction.
#[derive(Debug, Clone, Copy)]
pub struct ProviderRegistration {
    /// Provider kind matched against [`crate::ProviderInstanceConfig::kind`].
    kind: ProviderKindStatic,
    /// Constructor called for each configured provider instance of this kind.
    constructor: ProviderConstructor,
    /// Schema for this provider kind's full provider instance configuration object.
    instance_config_schema: ProviderInstanceConfigSchemaFn,
}

impl ProviderRegistration {
    /// Creates a provider registration from erased inventory function pointers.
    ///
    /// This is public only so [`crate::provider_registration!`] can expand in
    /// downstream crates. Provider crates should use [`crate::submit_provider!`]
    /// or [`crate::provider_registration!`] instead of calling this directly.
    #[doc(hidden)]
    pub const fn __from_erased(
        kind: ProviderKindStatic,
        constructor: ProviderConstructor,
        instance_config_schema: ProviderInstanceConfigSchemaFn,
    ) -> Self {
        Self {
            kind,
            constructor,
            instance_config_schema,
        }
    }
}

inventory::collect!(ProviderRegistration);

/// Provider instance configuration schema exposed by a provider catalog.
///
/// Each item combines a registered provider kind with the full JSON Schema for
/// one [`ProviderInstanceConfig`] object of that kind. The schema includes
/// sigma's common provider fields and places provider-owned settings under the
/// nested `config` property. This is intended for documentation generators and
/// config UIs; provider constructors remain the source of truth for runtime
/// validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderInstanceConfigSchema {
    /// Registered provider kind described by this schema.
    pub kind: ProviderKind,
    /// JSON Schema for one provider instance configuration object.
    pub schema: Value,
}

/// Catalog of provider constructors discovered from inventory.
///
/// Build code uses this catalog to turn [`crate::ProviderInstanceConfig`] values
/// into initialized [`ProviderDriver`] instances. The order of inventory entries
/// is not stable, so the catalog rejects duplicate provider kinds.
#[derive(Debug, Clone)]
pub struct ProviderCatalog {
    registrations: HashMap<ProviderKind, ProviderRegistration>,
}

impl ProviderCatalog {
    /// Builds a catalog from all linked inventory registrations.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::DuplicateProviderRegistration`] if two linked
    /// crates register the same provider kind.
    pub fn from_inventory() -> SigmaResult<Self> {
        Self::from_registrations(inventory::iter::<ProviderRegistration>.into_iter().copied())
    }

    /// Builds a catalog from explicit registrations.
    ///
    /// This is primarily useful for tests because inventory iteration order is
    /// intentionally unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::DuplicateProviderRegistration`] when more than one
    /// registration uses the same kind.
    pub fn from_registrations(
        registrations: impl IntoIterator<Item = ProviderRegistration>,
    ) -> SigmaResult<Self> {
        let mut catalog = HashMap::new();

        for registration in registrations {
            let kind = ProviderKind::from(registration.kind);
            if catalog.insert(kind.clone(), registration).is_some() {
                return Err(SigmaError::DuplicateProviderRegistration {
                    kind: kind.to_string(),
                });
            }
        }

        Ok(Self {
            registrations: catalog,
        })
    }

    /// Returns whether a provider kind is available in the catalog.
    pub fn contains_kind(&self, kind: &ProviderKind) -> bool {
        self.registrations.contains_key(kind)
    }

    /// Returns the constructor for a provider kind.
    pub fn get(&self, kind: &ProviderKind) -> Option<ProviderConstructor> {
        self.registrations
            .get(kind)
            .map(|registration| registration.constructor)
    }

    /// Returns full provider instance configuration schemas sorted by kind.
    ///
    /// Inventory iteration order is intentionally unspecified, so this method
    /// sorts by [`ProviderKind`] before returning schemas. Each schema describes
    /// one complete [`ProviderInstanceConfig`] shape for a specific kind,
    /// including sigma's common fields and that provider's nested `config`
    /// schema.
    pub fn provider_instance_config_schemas(&self) -> Vec<ProviderInstanceConfigSchema> {
        let mut registrations = self.registrations.iter().collect::<Vec<_>>();
        registrations.sort_by_key(|(kind, _)| *kind);

        registrations
            .into_iter()
            .map(|(kind, registration)| ProviderInstanceConfigSchema {
                kind: kind.clone(),
                schema: (registration.instance_config_schema)(registration.kind),
            })
            .collect()
    }
}

/// Generates the full provider instance schema for a typed provider config.
///
/// This helper is used by [`crate::provider_registration!`] and is exposed so
/// macro expansion works from provider crates. The generated schema describes a
/// complete [`ProviderInstanceConfig`] object for `kind`, with common sigma
/// provider fields at the top level and `TConfig` under `config`.
#[doc(hidden)]
pub fn provider_instance_config_schema_for<TConfig>(kind: ProviderKindStatic) -> Value
where
    TConfig: JsonSchema,
{
    let mut schema = schema_value_for::<ProviderInstanceConfigSchemaView<TConfig>>();
    let object = schema
        .as_object_mut()
        .expect("schemars generated a non-object provider instance schema");

    object.insert(
        "title".to_string(),
        format!("{} provider instance", kind.as_str()).into(),
    );
    object.insert(
        "$schema".to_string(),
        "https://json-schema.org/draft/2020-12/schema".into(),
    );
    object.insert("additionalProperties".to_string(), false.into());

    let properties = object
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .expect("provider instance schema should have object properties");
    properties.insert(
        "kind".to_string(),
        json!({
            "const": kind.as_str(),
            "description": "Registered provider kind used to select the provider constructor."
        }),
    );

    schema
}

fn schema_value_for<T>() -> Value
where
    T: JsonSchema,
{
    SchemaSettings::draft2020_12()
        .with(|settings| {
            settings.inline_subschemas = true;
        })
        .into_generator()
        .into_root_schema_for::<T>()
        .to_value()
}

#[derive(JsonSchema)]
#[schemars(deny_unknown_fields)]
#[allow(dead_code)]
struct ProviderInstanceConfigSchemaView<TConfig: JsonSchema> {
    /// Stable provider instance id used by deployments and direct provider-model routing.
    id: ProviderId,
    /// Registered provider kind used to select the provider constructor.
    kind: ProviderKind,
    #[serde(flatten)]
    common: ProviderCommonConfigSchemaView,
    /// Provider-specific configuration passed to the selected provider constructor.
    #[serde(default)]
    #[schemars(!default)]
    config: TConfig,
}

#[derive(Default, JsonSchema)]
#[schemars(deny_unknown_fields)]
#[allow(dead_code)]
struct ProviderCommonConfigSchemaView {
    /// Optional provider API base URL override.
    #[serde(default)]
    #[schemars(!default)]
    api_base: String,
    /// Optional provider credential.
    #[serde(default)]
    #[schemars(!default)]
    api_key: String,
    /// Static headers made available to the provider constructor.
    #[serde(default)]
    headers: HashMap<String, String>,
}

/// Initialization data passed to a provider constructor.
///
/// sigma creates one `ProviderInit` per [`crate::ProviderInstanceConfig`] when
/// [`crate::Client::build`] runs the catalog. Provider drivers should validate
/// any provider-specific config here, resolve credentials from environment
/// fallbacks, and return [`SigmaError::ProviderConfig`] for invalid input.
///
/// `TConfig` is `ProviderConfigMap` for the raw inventory entry point and the
/// typed `config` struct for downstream constructors invoked through
/// [`crate::submit_provider!`]. Use [`ProviderInit::deserialize_config`] or
/// [`ProviderInit::into_typed_config`] when starting from raw configuration.
#[derive(Debug, Clone)]
pub struct ProviderInit<TConfig = ProviderConfigMap> {
    /// Configured provider instance id.
    pub id: ProviderId,
    /// Runtime provider kind that matched this constructor.
    pub kind: ProviderKind,
    /// Common provider configuration fields.
    pub common: ProviderCommonConfig,
    /// Provider-specific configuration from the nested `config` object.
    ///
    /// Provider constructors registered with [`crate::submit_provider!`] receive
    /// their typed configuration here. Raw `ProviderInit` values created from
    /// [`ProviderInstanceConfig`] keep this as [`ProviderConfigMap`] until the
    /// registration wrapper deserializes it.
    pub config: TConfig,
}

impl ProviderInit<ProviderConfigMap> {
    /// Deserializes the provider-specific `config` object into a typed value.
    ///
    /// Provider constructors should define their own configuration structs,
    /// usually with Serde defaults and `deny_unknown_fields`. Provider
    /// registrations created with [`crate::submit_provider!`] call this through
    /// the generated wrapper before invoking the typed constructor. This method
    /// remains available for tests and low-level integration code that starts
    /// from raw [`ProviderConfigMap`] values.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::ProviderConfig`] when the nested `config` object
    /// does not match `T`.
    pub fn deserialize_config<T>(&self) -> SigmaResult<T>
    where
        T: DeserializeOwned,
    {
        serde_json::from_value(Value::Object(self.config.clone())).map_err(|err| {
            SigmaError::ProviderConfig {
                provider: Some(self.id.clone()),
                message: format!("invalid provider config: {err}"),
            }
        })
    }

    /// Converts this initialization value to one with typed provider config.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::ProviderConfig`] when the nested `config` object
    /// does not match `T`.
    pub fn into_typed_config<T>(self) -> SigmaResult<ProviderInit<T>>
    where
        T: DeserializeOwned,
    {
        let config = self.deserialize_config()?;

        Ok(ProviderInit {
            id: self.id,
            kind: self.kind,
            common: self.common,
            config,
        })
    }
}

impl From<ProviderInstanceConfig> for ProviderInit<ProviderConfigMap> {
    fn from(value: ProviderInstanceConfig) -> Self {
        Self {
            id: value.id,
            kind: value.kind,
            common: value.common,
            config: value.config,
        }
    }
}

/// Initialized provider instance.
///
/// A driver represents one configured provider instance, including its
/// credentials, base URL, and provider-specific config. Capabilities are
/// exposed through optional methods such as [`ProviderDriver::chat`].
pub trait ProviderDriver: Send + Sync {
    /// Returns the configured provider instance id.
    fn id(&self) -> &ProviderId;

    /// Returns the provider kind used to create this instance.
    fn kind(&self) -> &ProviderKind;

    /// Returns the standard chat adapter when the provider supports chat.
    fn chat(&self) -> Option<&dyn ChatCompletionAdapter> {
        None
    }

    /// Returns a custom chat handler that bypasses the generic HTTP adapter flow.
    ///
    /// This is intended for providers that cannot be modeled as
    /// messages-to-HTTP-transform-to-response-transform.
    fn custom_chat(&self) -> Option<&dyn CustomChatProvider> {
        None
    }
}

/// Fully routed chat request passed to a [`CustomChatProvider`].
#[derive(Debug, Clone, Copy)]
pub struct RoutedChatRequest<'a> {
    /// Provider instance selected for the request.
    pub provider: &'a ProviderId,
    /// Deployment selected for the request, if routing used one.
    pub deployment: Option<&'a DeploymentId>,
    /// Public model name requested by the caller or deployment.
    pub public_model: &'a ModelName,
    /// Provider-native model name to use.
    pub provider_model: &'a ModelName,
    /// Original chat completion request.
    pub request: &'a ChatRequest,
    /// Opaque model metadata from the selected deployment, when routing used one.
    pub model_info: Option<&'a Value>,
}

/// Optional provider capability for fully custom chat handling.
///
/// Implement this when a provider needs to bypass sigma's generic adapter and
/// HTTP execution pipeline. Most HTTP JSON providers should implement
/// [`ChatCompletionAdapter`] instead. Routed request fields are borrowed for
/// the duration of the call; custom providers should clone only the values they
/// must keep beyond the returned future or stream construction.
#[async_trait]
pub trait CustomChatProvider: Send + Sync {
    /// Creates one chat completion through provider-specific code.
    async fn create(&self, request: RoutedChatRequest<'_>) -> SigmaResult<ChatResponse>;

    /// Creates a streaming chat completion through provider-specific code.
    async fn create_stream(&self, request: RoutedChatRequest<'_>) -> SigmaResult<ChatStream>;
}

/// Routing metadata shared across chat adapter request and response hooks.
///
/// The context identifies the provider instance, selected deployment, public
/// model name, provider-native model name, and deployment model metadata for a
/// single routed chat request. Adapters receive it again when transforming
/// regular responses or provider byte streams so parsing can depend on the same
/// routing state used to build the request.
#[derive(Debug, Clone)]
pub struct ChatAdapterContext<'a> {
    /// Provider instance selected for the request.
    pub provider: &'a ProviderId,
    /// Deployment selected for the request, if routing used one.
    pub deployment: Option<&'a DeploymentId>,
    /// Public model name requested by the caller or deployment.
    pub public_model: &'a ModelName,
    /// Provider-native model name to send to the provider.
    pub provider_model: &'a ModelName,
    /// Opaque model metadata from the selected deployment, when routing used one.
    pub model_info: Option<&'a Value>,
    /// Request-scoped, type-erased provider state created while transforming
    /// the outbound request.
    ///
    /// This is never serialized. Adapters can use it when a response transform
    /// needs data derived from the request, such as Anthropic tool-name maps.
    /// Use [`ChatAdapterContext::provider_state_as`] to recover the original
    /// typed value.
    pub provider_state: Option<ProviderState>,
}

impl ChatAdapterContext<'_> {
    /// Returns a reference to the provider state if it was set with type `T`.
    ///
    /// Returns `None` when the request did not produce any state, or when the
    /// state was set with a different concrete type. Adapters should pick a
    /// stable concrete type per provider so request and response transforms
    /// agree on the shape they share.
    pub fn provider_state_as<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.provider_state
            .as_ref()
            .and_then(|state| state.downcast_ref::<T>())
    }
}

/// Provider-neutral request data passed through the chat adapter lifecycle.
///
/// The adapter owns the full `ChatRequestParams → native body` translation:
/// it merges deployment defaults with the request's typed chat parameters,
/// applies adapter-owned compatibility rules, and finally builds a structured
/// JSON provider request. sigma's client layer only performs routing and
/// dispatch.
#[derive(Debug, Clone)]
pub struct ChatAdapterRequest<'a> {
    /// Routing metadata for this adapter call.
    pub context: ChatAdapterContext<'a>,
    /// Original provider-neutral chat request.
    ///
    /// Adapters read messages from `request.messages`, typed chat parameters
    /// from `request.params`, and provider-scoped overrides from
    /// `request.provider_options.get(provider_id)`.
    pub request: &'a ChatRequest,
    /// Deployment-level chat parameter defaults to merge before request
    /// parameters.
    ///
    /// `None` when the resolved route does not come from a configured
    /// deployment (for example, [`crate::ModelRef::ProviderModel`] direct
    /// routing).
    pub deployment_defaults: Option<&'a ChatParameterMap>,
    /// Whether the caller invoked `create_stream`.
    ///
    /// Adapters use this to select streaming endpoints (for example Gemini's
    /// `streamGenerateContent`) and to inject provider-native streaming request
    /// fields such as `"stream": true`.
    pub streaming: bool,
}

/// Stream of semantic chat chunks returned by sigma.
///
/// Each item is either one [`crate::types::chat::ChatStreamChunk`] or a
/// [`SigmaError`]. Errors are non-fatal at the type level — callers may keep
/// polling after a failed item — but most provider adapters terminate the
/// stream after surfacing one error because the underlying HTTP body is no
/// longer trustworthy. Adapters produce these streams from
/// [`ChatCompletionAdapter::transform_stream`] and sigma forwards them
/// unchanged through [`crate::Client::create_stream`].
pub type ChatStream = Pin<Box<dyn Stream<Item = SigmaResult<ChatStreamChunk>> + Send + 'static>>;

/// Provider adapter for the generic chat HTTP pipeline.
///
/// The lifecycle for `create` is:
///
/// 1. [`ChatCompletionAdapter::endpoint`]
/// 2. [`ChatCompletionAdapter::transform_request`] — owns the full
///    `ChatRequestParams` to native body translation, including merging
///    deployment defaults and applying adapter-owned compatibility rules,
///    provider-specific renames, or post-processing
/// 3. [`ChatCompletionAdapter::sign_request`]
/// 4. sigma sends the signed request with its shared [`reqwest::Client`]
/// 5. non-success HTTP statuses use
///    [`ChatCompletionAdapter::transform_error_response`]
/// 6. success responses use [`ChatCompletionAdapter::transform_response`]
///
/// `create_stream` follows the same preparation path. Streams use
/// [`ChatCompletionAdapter::transform_error_response`] for non-success HTTP
/// statuses or [`ChatCompletionAdapter::transform_stream`] to convert
/// successful provider bytes into semantic chat chunks.
pub trait ChatCompletionAdapter: Send + Sync {
    /// Selects the provider endpoint for a prepared chat request.
    fn endpoint(&self, request: &ChatAdapterRequest<'_>) -> SigmaResult<ProviderEndpoint>;

    /// Builds a structured provider HTTP request from prepared chat inputs.
    fn transform_request(
        &self,
        request: ChatAdapterRequest<'_>,
        endpoint: ProviderEndpoint,
    ) -> SigmaResult<ProviderRequest>;

    /// Signs or authenticates the provider request.
    fn sign_request(&self, request: ProviderRequest) -> SigmaResult<SignedProviderRequest>;

    /// Transforms a completed provider HTTP response into a chat response.
    ///
    /// This hook is synchronous by design. The shared [`reqwest::Client`] owned
    /// by [`crate::Client`] has already awaited and buffered the response body.
    /// Providers that need asynchronous response handling should implement
    /// [`CustomChatProvider`] instead of the generic adapter.
    fn transform_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaResult<ChatResponse>;

    /// Transforms a non-success provider HTTP response into a sigma error.
    ///
    /// sigma calls this hook when the provider returns an HTTP status outside
    /// the 2xx success range. The full response body has been buffered so
    /// adapters can parse provider-native error JSON and return
    /// [`SigmaError::ProviderBusiness`] with stable codes, human-readable
    /// messages, and structured details. Transport failures that do not produce
    /// an HTTP response still return [`SigmaError::Http`].
    ///
    /// The default implementation only inspects the HTTP status and returns the
    /// raw response body as a UTF-8 lossy string in
    /// [`SigmaError::ProviderBusiness`]'s `message`, with `code` and `details`
    /// unset. Real adapters should override this to parse provider-specific
    /// error envelopes, surface stable codes, and—when applicable—return one of
    /// the semantic variants such as `SigmaError::RateLimited` or
    /// `SigmaError::AuthFailed` so callers can implement portable retry or
    /// fallback logic.
    fn transform_error_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaError {
        let message = if response.body.is_empty() {
            response
                .status
                .canonical_reason()
                .unwrap_or("provider returned unsuccessful HTTP status")
                .to_string()
        } else {
            String::from_utf8_lossy(&response.body).into_owned()
        };

        SigmaError::ProviderBusiness {
            provider: context.provider.to_owned(),
            status: response.status,
            code: None,
            message,
            details: None,
        }
    }

    /// Transforms provider streaming bytes into semantic chat chunks.
    ///
    /// The returned stream may synchronously parse, filter, or reshape each raw
    /// provider byte frame. Network polling remains owned by sigma's HTTP
    /// execution path; this hook only defines provider-specific chunk
    /// translation. Providers that need fully custom asynchronous stream
    /// execution should implement [`CustomChatProvider`].
    fn transform_stream(
        &self,
        context: &ChatAdapterContext<'_>,
        stream: ProviderByteStream,
    ) -> SigmaResult<ChatStream>;
}

pub(crate) fn deployment_model_info(deployment: Option<&ModelDeploymentConfig>) -> Option<&Value> {
    deployment.map(|deployment| &deployment.model_info)
}

/// Creates a typed provider registration value.
///
/// This expression macro is useful in tests or custom catalogs where callers
/// want deterministic registration lists without using global inventory. The
/// `config` type must implement [`serde::Deserialize`] and
/// [`schemars::JsonSchema`]. sigma deserializes
/// [`ProviderInstanceConfig::config`] into that type before calling the
/// constructor, and uses the same type to generate the provider config schema.
///
/// ```ignore
/// # use std::sync::Arc;
/// # use schemars::JsonSchema;
/// # use serde::Deserialize;
/// # use sigma::{
/// #     ProviderDriver, ProviderInit, ProviderKindStatic, SigmaResult, provider_registration,
/// # };
/// #[derive(Debug, Default, Deserialize, JsonSchema)]
/// #[serde(default, deny_unknown_fields)]
/// struct MyProviderConfig {
///     timeout_ms: Option<u64>,
/// }
///
/// fn from_config(
///     init: ProviderInit<MyProviderConfig>,
/// ) -> SigmaResult<Arc<dyn ProviderDriver>> {
///     # let _ = init;
///     # todo!()
/// }
///
/// let registration = provider_registration! {
///     kind: ProviderKindStatic::new("my-provider"),
///     constructor: from_config,
///     config: MyProviderConfig,
/// };
/// # let _ = registration;
/// ```
#[macro_export]
macro_rules! provider_registration {
    (kind: $kind:expr, constructor: $constructor:path, config: $config:ty $(,)?) => {{
        fn __sigma_provider_constructor(
            init: $crate::ProviderInit,
        ) -> $crate::SigmaResult<::std::sync::Arc<dyn $crate::ProviderDriver>> {
            let init = init.into_typed_config::<$config>()?;
            $constructor(init)
        }

        fn __sigma_provider_instance_config_schema(
            kind: $crate::ProviderKindStatic,
        ) -> $crate::__private::serde_json::Value {
            $crate::provider_instance_config_schema_for::<$config>(kind)
        }

        $crate::ProviderRegistration::__from_erased(
            $kind,
            __sigma_provider_constructor,
            __sigma_provider_instance_config_schema,
        )
    }};
}

/// Registers a typed provider constructor in sigma's distributed inventory.
///
/// Call this macro once per provider kind from the provider crate or module.
/// The constructor is called once for each matching
/// [`ProviderInstanceConfig`] during [`Client::build`](crate::Client::build).
/// The provider-specific `config` type is also used to generate the nested
/// provider configuration schema exposed by [`ProviderCatalog`].
///
/// ```ignore
/// use std::sync::Arc;
/// use schemars::JsonSchema;
/// use serde::Deserialize;
/// use sigma::{
///     ProviderDriver, ProviderInit, ProviderKindStatic, SigmaResult, submit_provider,
/// };
///
/// struct MyProvider;
///
/// #[derive(Debug, Default, Deserialize, JsonSchema)]
/// #[serde(default, deny_unknown_fields)]
/// struct MyProviderConfig {
///     /// Provider-specific request timeout in milliseconds.
///     timeout_ms: Option<u64>,
/// }
///
/// impl MyProvider {
///     fn from_config(
///         init: ProviderInit<MyProviderConfig>,
///     ) -> SigmaResult<Arc<dyn ProviderDriver>> {
///         // Validate init.config, credentials, and endpoint overrides here.
///         todo!()
///     }
/// }
///
/// submit_provider! {
///     kind: ProviderKindStatic::new("my-provider"),
///     constructor: MyProvider::from_config,
///     config: MyProviderConfig,
/// }
/// ```
#[macro_export]
macro_rules! submit_provider {
    (kind: $kind:expr, constructor: $constructor:path, config: $config:ty $(,)?) => {
        $crate::inventory::submit! {
            $crate::provider_registration! {
                kind: $kind,
                constructor: $constructor,
                config: $config,
            }
        }
    };
}
