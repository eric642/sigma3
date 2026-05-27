//! Shared OpenAI-family chat body builder.
//!
//! Multiple providers share the OpenAI request shape but customize a few
//! semantic fields. This module defines a `pub(crate)` template-method trait
//! that owns the canonical build pipeline:
//!
//! 1. Merge deployment defaults with the request's typed chat parameters,
//!    optionally injecting the streaming flag.
//! 2. Resolve and apply caller-configured chat parameter rules
//!    (drop / unsupported policy / rename / nested drop).
//! 3. Run the adapter's `post_process_params` hook for provider-specific
//!    renames.
//! 4. Build the OpenAI-style JSON body via [`openai_chat_body`].
//! 5. Run the adapter's `post_process_body` hook for any final body shape
//!    mutations.
//!
//! Providers in the OpenAI family implement [`OpenAiChatBodyBuilder`] and call
//! [`OpenAiChatBodyBuilder::build_chat_body`] from inside their
//! `ChatCompletionAdapter::transform_request`. Adding a new family member
//! (e.g. Moonshot, DeepSeek) is a matter of overriding the two hooks; sigma's
//! client layer never has to be updated.

use serde_json::{Map, Value};

use crate::config::{ChatParamConfig, ChatParameterMap};
use crate::providers::chat_params::{
    apply_chat_param_rules, merge_chat_params, resolve_chat_param_rules,
};
use crate::types::chat::{ChatMessage, ChatRequest};
use crate::{ModelName, ProviderId, SigmaResult};

use super::request::openai_chat_body;

/// Inputs the OpenAI body pipeline forwards to the adapter's hooks.
pub(crate) struct OpenAiBuildContext<'a> {
    pub provider: &'a ProviderId,
    pub provider_model: &'a ModelName,
    pub messages: &'a [ChatMessage],
    pub provider_options: Option<&'a ChatParameterMap>,
    pub streaming: bool,
}

/// Template-method trait that codifies the OpenAI-family chat body pipeline.
///
/// Concrete adapters (the OpenAI provider, OpenAI-compatible providers,
/// future Moonshot/DeepSeek/Qwen drivers) implement this trait alongside
/// [`crate::ChatCompletionAdapter`]. They override the two hooks
/// (`post_process_params`, `post_process_body`) for provider-specific
/// behavior and call [`OpenAiChatBodyBuilder::build_chat_body`] from inside
/// their `transform_request`.
pub(crate) trait OpenAiChatBodyBuilder {
    /// Identity of this adapter; used for error attribution and lookups.
    fn provider_id(&self) -> &ProviderId;

    /// Built-in chat parameter support set for this adapter.
    fn default_supported_chat_params(&self) -> &'static [&'static str];

    /// Whether this adapter wants the `"stream": true` body field injected
    /// before chat parameter rule validation when sigma intends to stream.
    fn inject_stream_flag(&self) -> bool {
        true
    }

    /// Provider-specific top-level renames applied AFTER caller-configured
    /// rename rules. Default: identity.
    fn post_process_params(&self, _params: &mut ChatParameterMap) {}

    /// Final body mutation hook applied AFTER the OpenAI body has been
    /// assembled (including provider option merge). Default: identity.
    fn post_process_body(
        &self,
        _ctx: &OpenAiBuildContext<'_>,
        _body: &mut Map<String, Value>,
    ) -> SigmaResult<()> {
        Ok(())
    }

    /// Canonical OpenAI-family build pipeline.
    fn build_chat_body(
        &self,
        ctx: &OpenAiBuildContext<'_>,
        request: &ChatRequest,
        deployment_defaults: Option<&ChatParameterMap>,
        chat_param_config: Option<&ChatParamConfig>,
    ) -> SigmaResult<Value> {
        let inject_stream = ctx.streaming && self.inject_stream_flag();
        let mut params = merge_chat_params(deployment_defaults, request, inject_stream)?;

        let rules = resolve_chat_param_rules(
            self.default_supported_chat_params(),
            chat_param_config,
            ctx.provider_model,
        );
        apply_chat_param_rules(self.provider_id(), &mut params, &rules)?;

        self.post_process_params(&mut params);

        let body = openai_chat_body(
            ctx.provider,
            ctx.provider_model,
            ctx.messages,
            &params,
            ctx.provider_options,
        )?;
        let Value::Object(mut body) = body else {
            unreachable!("openai_chat_body always returns a JSON object");
        };
        self.post_process_body(ctx, &mut body)?;
        Ok(Value::Object(body))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use serde_json::json;

    use super::*;
    use crate::ModelRef;
    use crate::types::chat::{ChatRequest, ChatRequestParams, UserMessage};

    const SUPPORTED: &[&str] = &["temperature", "stream"];

    struct StubAdapter {
        provider: ProviderId,
        post_process_params_called: Cell<bool>,
        post_process_body_called: Cell<bool>,
    }

    impl StubAdapter {
        fn new() -> Self {
            Self {
                provider: ProviderId::from("stub"),
                post_process_params_called: Cell::new(false),
                post_process_body_called: Cell::new(false),
            }
        }
    }

    impl OpenAiChatBodyBuilder for StubAdapter {
        fn provider_id(&self) -> &ProviderId {
            &self.provider
        }
        fn default_supported_chat_params(&self) -> &'static [&'static str] {
            SUPPORTED
        }
        fn post_process_params(&self, params: &mut ChatParameterMap) {
            self.post_process_params_called.set(true);
            params.insert("__post_param_sentinel".to_string(), Value::Bool(true));
        }
        fn post_process_body(
            &self,
            _ctx: &OpenAiBuildContext<'_>,
            body: &mut Map<String, Value>,
        ) -> SigmaResult<()> {
            self.post_process_body_called.set(true);
            body.insert("__post_body_sentinel".to_string(), Value::Bool(true));
            Ok(())
        }
    }

    fn sample_request() -> ChatRequest {
        ChatRequest::new(
            ModelRef::model("public-model"),
            vec![UserMessage::text("hi").into()],
        )
        .with_params(ChatRequestParams {
            temperature: Some(0.5),
            ..Default::default()
        })
    }

    #[test]
    fn build_chat_body_runs_hooks_in_order_with_streaming_inject() {
        let adapter = StubAdapter::new();
        let request = sample_request();
        let model = ModelName::from("native-model");
        let ctx = OpenAiBuildContext {
            provider: adapter.provider_id(),
            provider_model: &model,
            messages: &request.messages,
            provider_options: None,
            streaming: true,
        };

        let body = adapter
            .build_chat_body(&ctx, &request, None, None)
            .expect("body builds");

        assert!(adapter.post_process_params_called.get());
        assert!(adapter.post_process_body_called.get());

        let body = body.as_object().unwrap();
        assert_eq!(body.get("model"), Some(&json!("native-model")));
        assert_eq!(body.get("temperature"), Some(&json!(0.5_f32)));
        assert_eq!(body.get("stream"), Some(&json!(true)));
        assert_eq!(body.get("__post_param_sentinel"), Some(&json!(true)));
        assert_eq!(body.get("__post_body_sentinel"), Some(&json!(true)));
        assert!(body.contains_key("messages"));
    }

    #[test]
    fn build_chat_body_skips_stream_inject_when_not_streaming() {
        let adapter = StubAdapter::new();
        let request = sample_request();
        let model = ModelName::from("native-model");
        let ctx = OpenAiBuildContext {
            provider: adapter.provider_id(),
            provider_model: &model,
            messages: &request.messages,
            provider_options: None,
            streaming: false,
        };

        let body = adapter
            .build_chat_body(&ctx, &request, None, None)
            .expect("body builds");
        assert!(body.as_object().unwrap().get("stream").is_none());
    }

    #[test]
    fn build_chat_body_propagates_unsupported_param_error() {
        let adapter = StubAdapter::new();
        let request = sample_request().with_params(ChatRequestParams {
            temperature: Some(0.5),
            top_p: Some(0.9),
            ..Default::default()
        });
        let model = ModelName::from("native-model");
        let ctx = OpenAiBuildContext {
            provider: adapter.provider_id(),
            provider_model: &model,
            messages: &request.messages,
            provider_options: None,
            streaming: false,
        };

        let err = adapter
            .build_chat_body(&ctx, &request, None, None)
            .unwrap_err();
        match err {
            crate::SigmaError::UnsupportedParams { params, .. } => {
                assert!(params.contains(&"top_p".to_string()));
            }
            other => panic!("expected UnsupportedParams, got {other:?}"),
        }
        assert!(!adapter.post_process_params_called.get());
        assert!(!adapter.post_process_body_called.get());
    }
}
