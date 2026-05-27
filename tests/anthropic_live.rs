#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::future::Future;
use std::time::Duration;

use futures_util::StreamExt;
use http::StatusCode;
use serde_json::{Value, json};
use sigma::types::chat::{
    CacheControl, ChatMessage, ChatRequest, ChatRequestParams, ChatResponse, FunctionTool,
    NamedFunctionToolChoice, StreamOptions, TextPart, ToolCall, ToolChoice, ToolDefinition,
    UserContent, UserContentPart, UserMessage,
};
use sigma::types::shared::{FunctionName, FunctionObject, ReasoningEffort};
use sigma::{
    ChatParameterMap, ChatStream, Client, ClientConfig, ModelDeploymentConfig, ModelName, ModelRef,
    ProviderCommonConfig, ProviderConfigMap, ProviderId, ProviderInstanceConfig, ProviderKind,
    SecretString, SigmaError, SigmaResult,
};

const DEFAULT_ANTHROPIC_TEST_MODEL: &str = "claude-haiku-4-5-20251001";
const DEFAULT_ANTHROPIC_BAD_MODEL: &str = "sigma-nonexistent-anthropic-model";
const DEFAULT_MAX_TOKENS: u32 = 32;
const PROMPT_CACHE_ATTEMPTS: u8 = 3;
const ANTHROPIC_PROVIDER_ID: &str = "anthropic-live";
const ANTHROPIC_PUBLIC_MODEL: &str = "anthropic-live-model";
const TOOL_FUNCTION_NAME: &str = "lookup_city_weather";

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiveAnthropicConfig {
    api_key: String,
    api_base: Option<String>,
    model: String,
    bad_model: String,
    max_tokens: u32,
}

#[derive(Debug, Clone)]
struct LiveEnv {
    process: HashMap<String, String>,
    dotenv: HashMap<String, String>,
}

impl LiveEnv {
    fn load() -> Self {
        let dotenv = fs::read_to_string(".env")
            .map(|contents| dotenv_values(&contents))
            .unwrap_or_default();

        Self {
            process: env::vars().collect(),
            dotenv,
        }
    }

    fn from_sources(process: HashMap<String, String>, dotenv: &str) -> Self {
        Self {
            process,
            dotenv: dotenv_values(dotenv),
        }
    }

    fn value(&self, name: &str) -> Option<String> {
        self.process
            .get(name)
            .or_else(|| self.dotenv.get(name))
            .filter(|value| !value.trim().is_empty())
            .cloned()
    }

    fn first_value(&self, names: &[&str]) -> Option<String> {
        names.iter().find_map(|name| self.value(name))
    }

    fn anthropic_config(&self) -> Option<LiveAnthropicConfig> {
        let api_key = self.value("ANTHROPIC_API_KEY")?;
        let api_base = self.first_value(&["ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"]);
        let model = self
            .first_value(&["SIGMA_ANTHROPIC_TEST_MODEL", "ANTHROPIC_MODEL"])
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_TEST_MODEL.to_string());
        let bad_model = self
            .value("SIGMA_ANTHROPIC_TEST_BAD_MODEL")
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_BAD_MODEL.to_string());
        let max_tokens = self
            .value("SIGMA_ANTHROPIC_TEST_MAX_TOKENS")
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        Some(LiveAnthropicConfig {
            api_key,
            api_base,
            model,
            bad_model,
            max_tokens,
        })
    }
}

fn dotenv_values(contents: &str) -> HashMap<String, String> {
    contents
        .lines()
        .filter_map(dotenv_pair)
        .collect::<HashMap<_, _>>()
}

fn dotenv_pair(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
    let (key, value) = line.split_once('=')?;
    let key = key.trim();
    if !is_dotenv_key(key) {
        return None;
    }

    Some((key.to_string(), dotenv_value(value)))
}

fn is_dotenv_key(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn dotenv_value(value: &str) -> String {
    let value = value.trim();
    if let Some(value) = quoted_dotenv_value(value, '"') {
        return value;
    }
    if let Some(value) = quoted_dotenv_value(value, '\'') {
        return value;
    }

    value
        .split_once(" #")
        .map_or(value, |(value, _)| value)
        .trim_end()
        .to_string()
}

fn quoted_dotenv_value(value: &str, quote: char) -> Option<String> {
    let value = value.strip_prefix(quote)?;
    let index = value.find(quote)?;
    Some(value[..index].to_string())
}

fn live_client_config(config: &LiveAnthropicConfig) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: ProviderId::from(ANTHROPIC_PROVIDER_ID),
            kind: ProviderKind::from("anthropic"),
            common: ProviderCommonConfig {
                api_base: config.api_base.clone(),
                api_key: Some(SecretString::from(config.api_key.clone())),
                headers: HashMap::new(),
            },
            config: ProviderConfigMap::new(),
        }],
        deployments: vec![ModelDeploymentConfig {
            id: "anthropic-live-chat".into(),
            public_model: ModelName::from(ANTHROPIC_PUBLIC_MODEL),
            provider: ProviderId::from(ANTHROPIC_PROVIDER_ID),
            provider_model: ModelName::from(config.model.clone()),
            defaults: serde_json::Map::new(),
            model_info: serde_json::Value::Null,
        }],
        default_model: Some(ModelName::from(ANTHROPIC_PUBLIC_MODEL)),
    }
}

fn live_setup() -> SigmaResult<Option<(Client, LiveAnthropicConfig)>> {
    let Some(config) = LiveEnv::load().anthropic_config() else {
        eprintln!(
            "ANTHROPIC_API_KEY is not set in the environment or .env; skipping live Anthropic smoke test"
        );
        return Ok(None);
    };

    let client = Client::build(live_client_config(&config))?;
    Ok(Some((client, config)))
}

fn user_text_message(content: &str) -> ChatMessage {
    ChatMessage::User(UserMessage::from(content))
}

fn request_with(
    messages: Vec<ChatMessage>,
    model: ModelRef,
    max_tokens: u32,
    update_params: impl FnOnce(&mut ChatRequestParams),
) -> ChatRequest {
    let mut params = ChatRequestParams {
        max_completion_tokens: Some(max_tokens),
        ..Default::default()
    };
    update_params(&mut params);

    ChatRequest::new(model, messages).with_params(params)
}

fn live_request(max_tokens: u32) -> ChatRequest {
    request_with(
        vec![user_text_message("Reply with the single word pong.")],
        ModelRef::model(ANTHROPIC_PUBLIC_MODEL),
        max_tokens,
        |_| {},
    )
}

fn full_stream_usage_request(config: &LiveAnthropicConfig) -> ChatRequest {
    request_with(
        vec![user_text_message("Reply exactly with sigma-stream-pong.")],
        ModelRef::model(ANTHROPIC_PUBLIC_MODEL),
        config.max_tokens.max(16),
        |params| {
            params.stream_options = Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
        },
    )
}

fn function_tool_request(config: &LiveAnthropicConfig) -> ChatRequest {
    request_with(
        vec![user_text_message(
            "Use the weather lookup tool for city San Francisco.",
        )],
        ModelRef::model(ANTHROPIC_PUBLIC_MODEL),
        config.max_tokens.max(128),
        configure_function_tool,
    )
}

fn stream_function_tool_request(config: &LiveAnthropicConfig) -> ChatRequest {
    request_with(
        vec![user_text_message(
            "Use the weather lookup tool for city San Francisco.",
        )],
        ModelRef::model(ANTHROPIC_PUBLIC_MODEL),
        config.max_tokens.max(128),
        |params| {
            configure_function_tool(params);
            params.stream_options = Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
        },
    )
}

fn configure_function_tool(params: &mut ChatRequestParams) {
    params.tools = Some(function_tools());
    params.tool_choice = Some(ToolChoice::Function(NamedFunctionToolChoice {
        function: FunctionName {
            name: TOOL_FUNCTION_NAME.to_string(),
        },
    }));
}

fn function_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition::Function(FunctionTool {
        function: FunctionObject {
            name: TOOL_FUNCTION_NAME.to_string(),
            description: Some("Look up the current weather for a city.".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to look up."
                    }
                },
                "required": ["city"],
                "additionalProperties": false
            })),
            strict: None,
        },
    })]
}

fn structured_output_request(config: &LiveAnthropicConfig) -> ChatRequest {
    request_with(
        vec![user_text_message(
            "Classify this text: 'This product is amazing!' Return only the requested JSON.",
        )],
        ModelRef::model(ANTHROPIC_PUBLIC_MODEL),
        config.max_tokens.max(128),
        |params| {
            params.response_format = Some(sigma::types::shared::ResponseFormat::JsonSchema {
                json_schema: sigma::types::shared::ResponseFormatJsonSchema {
                    name: "SentimentResult".to_string(),
                    description: Some("Sentiment classification result.".to_string()),
                    schema: Some(sentiment_schema()),
                    strict: Some(true),
                },
            });
        },
    )
}

fn provider_options_output_format_request(config: &LiveAnthropicConfig) -> ChatRequest {
    let mut request = request_with(
        vec![user_text_message(
            "Classify this text: 'This product is amazing!' Return only the requested JSON.",
        )],
        ModelRef::model(ANTHROPIC_PUBLIC_MODEL),
        config.max_tokens.max(128),
        |_| {},
    );
    let mut options = ChatParameterMap::new();
    options.insert(
        "output_format".to_string(),
        json!({
            "type": "json_schema",
            "schema": sentiment_schema()
        }),
    );
    request
        .provider_options
        .insert(ProviderId::from(ANTHROPIC_PROVIDER_ID), options);
    request
}

fn sentiment_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            }
        },
        "required": ["sentiment"],
        "additionalProperties": false
    })
}

fn reasoning_effort_request(config: &LiveAnthropicConfig) -> ChatRequest {
    request_with(
        vec![user_text_message("Reply with a single short sentence.")],
        ModelRef::model(ANTHROPIC_PUBLIC_MODEL),
        config.max_tokens.max(1100),
        |params| {
            params.reasoning_effort = Some(ReasoningEffort::Low);
        },
    )
}

fn prompt_cache_request() -> ChatRequest {
    request_with(
        vec![ChatMessage::User(UserMessage {
            content: UserContent::Parts(vec![
                UserContentPart::Text(
                    TextPart::new(prompt_cache_prompt())
                        .with_cache_control(CacheControl::ephemeral()),
                ),
                UserContentPart::Text(TextPart::new(
                    "What are the key payment terms? Reply briefly.",
                )),
            ]),
            name: None,
        })],
        ModelRef::model(ANTHROPIC_PUBLIC_MODEL),
        DEFAULT_MAX_TOKENS,
        |_| {},
    )
}

fn prompt_cache_prompt() -> String {
    let stable_prefix = (0..2600)
        .map(|index| format!("anthropic-cache-prefix-{index:04}"))
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        "{stable_prefix}\n\nThis stable prefix is intentionally long for Anthropic prompt cache testing. Payment is due within 30 days of invoice receipt."
    )
}

fn bad_model_request(config: &LiveAnthropicConfig) -> ChatRequest {
    request_with(
        vec![user_text_message(
            "This request should fail because the model is invalid.",
        )],
        ModelRef::provider_model(
            ProviderId::from(ANTHROPIC_PROVIDER_ID),
            config.bad_model.clone(),
        ),
        config.max_tokens,
        |_| {},
    )
}

fn assert_chat_response_shape(response: &ChatResponse) {
    assert_eq!(response.object, "chat.completion");
    assert!(!response.id.is_empty(), "expected response id");
    assert!(!response.model.is_empty(), "expected response model");
    assert!(
        !response.choices.is_empty(),
        "expected at least one chat completion choice"
    );
}

fn assert_text_response(response: &ChatResponse) {
    assert_chat_response_shape(response);
    let content = response.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or_else(|| panic!("expected live Anthropic response content"));
    assert!(
        !content.trim().is_empty(),
        "expected non-empty live Anthropic text"
    );
}

fn assert_token_usage(response: &ChatResponse) {
    let usage = response
        .usage
        .as_ref()
        .unwrap_or_else(|| panic!("expected live Anthropic response usage"));

    assert!(usage.prompt_tokens > 0, "expected prompt token count");
    assert!(
        usage.completion_tokens > 0,
        "expected completion token count"
    );
    assert!(usage.total_tokens > 0, "expected total token count");
    assert!(
        usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens,
        "expected total tokens to include prompt and completion tokens"
    );
}

async fn first_stream_chunk(mut stream: ChatStream) -> SigmaResult<()> {
    let Some(chunk) = stream.next().await else {
        panic!("expected live Anthropic stream to yield at least one chunk");
    };
    let _chunk = chunk?;
    Ok(())
}

async fn collect_stream_text_and_usage(
    mut stream: ChatStream,
) -> SigmaResult<(String, Option<u32>)> {
    let mut text = String::new();
    let mut saw_chunk = false;
    let mut total_tokens = None;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        saw_chunk = true;

        if let Some(usage) = chunk.usage {
            total_tokens = Some(usage.total_tokens);
        }

        for choice in chunk.choices {
            if let Some(content) = choice.delta.content {
                text.push_str(&content);
            }
        }
    }

    assert!(saw_chunk, "expected live Anthropic stream to yield chunks");
    Ok((text, total_tokens))
}

#[derive(Debug, Default)]
struct StreamFunctionToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    total_tokens: Option<u32>,
    saw_tool_call: bool,
}

async fn collect_stream_function_tool_call(
    mut stream: ChatStream,
) -> SigmaResult<StreamFunctionToolCall> {
    let mut result = StreamFunctionToolCall::default();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        if let Some(usage) = chunk.usage {
            result.total_tokens = Some(usage.total_tokens);
        }

        for choice in chunk.choices {
            let Some(tool_calls) = choice.delta.tool_calls else {
                continue;
            };

            for tool_call in tool_calls {
                result.saw_tool_call = true;

                if result.id.is_none()
                    && let Some(id) = tool_call.id.filter(|id| !id.is_empty())
                {
                    result.id = Some(id);
                }

                let Some(function) = tool_call.function else {
                    continue;
                };
                if result.name.is_none()
                    && let Some(name) = function.name.filter(|name| !name.is_empty())
                {
                    result.name = Some(name);
                }
                if let Some(arguments) = function.arguments {
                    result.arguments.push_str(&arguments);
                }
            }
        }
    }

    Ok(result)
}

fn assert_response_function_tool_call(response: &ChatResponse) {
    assert_chat_response_shape(response);

    let tool_calls = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap_or_else(|| panic!("expected live Anthropic response tool_calls"));
    let tool_call = tool_calls
        .first()
        .unwrap_or_else(|| panic!("expected at least one live Anthropic tool call"));

    match tool_call {
        ToolCall::Function(call) => {
            assert!(!call.id.is_empty(), "expected non-empty tool call id");
            assert_eq!(call.function.name, TOOL_FUNCTION_NAME);
            assert_tool_arguments_json(&call.function.arguments);
        }
        other => panic!("expected function tool call, got {other:?}"),
    }
}

fn assert_stream_function_tool_call(result: &StreamFunctionToolCall) {
    assert!(result.saw_tool_call, "expected streamed tool call chunks");
    assert!(
        result.id.as_ref().is_some_and(|id| !id.is_empty()),
        "expected streamed tool call id"
    );
    assert_eq!(result.name.as_deref(), Some(TOOL_FUNCTION_NAME));
    assert_tool_arguments_json(&result.arguments);
    assert!(
        result.total_tokens.is_some_and(|value| value > 0),
        "expected streamed tool call usage"
    );
}

fn assert_tool_arguments_json(arguments: &str) {
    assert!(!arguments.trim().is_empty(), "expected tool arguments");
    let value = serde_json::from_str::<Value>(arguments)
        .unwrap_or_else(|err| panic!("expected JSON object tool arguments: {err}; {arguments}"));
    assert!(value.is_object(), "expected object tool arguments");
    assert!(
        value.get("city").and_then(Value::as_str).is_some(),
        "expected city argument"
    );
}

fn assert_structured_output(response: &ChatResponse) {
    assert_chat_response_shape(response);
    let value = structured_output_value(response);
    let sentiment = value
        .get("sentiment")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("expected sentiment field in structured output: {value}"));
    assert!(
        matches!(sentiment, "positive" | "negative" | "neutral"),
        "unexpected sentiment value: {sentiment}"
    );
}

fn structured_output_value(response: &ChatResponse) -> Value {
    if let Some(content) = response.choices[0].message.content.as_deref()
        && let Ok(value) = serde_json::from_str::<Value>(content)
    {
        return value;
    }

    let tool_calls = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap_or_else(|| panic!("expected text JSON or tool-call JSON structured output"));
    let ToolCall::Function(call) = &tool_calls[0] else {
        panic!("expected function tool structured output");
    };
    serde_json::from_str::<Value>(&call.function.arguments).unwrap_or_else(|err| {
        panic!(
            "expected JSON object structured output arguments: {err}; {}",
            call.function.arguments
        )
    })
}

fn cache_creation_tokens(response: &ChatResponse) -> u32 {
    response
        .usage
        .as_ref()
        .and_then(|usage| usage.cache_creation_input_tokens)
        .unwrap_or(0)
}

fn cache_read_tokens(response: &ChatResponse) -> u32 {
    response
        .usage
        .as_ref()
        .and_then(|usage| usage.cache_read_input_tokens)
        .unwrap_or(0)
}

async fn live_feature<F, Fut>(feature: &'static str, f: F) -> SigmaResult<()>
where
    F: FnOnce(Client, LiveAnthropicConfig) -> Fut,
    Fut: Future<Output = SigmaResult<()>>,
{
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };

    match f(client, config).await {
        Ok(()) => Ok(()),
        Err(err) if is_tolerated_live_feature_error(&err) => {
            eprintln!(
                "live Anthropic feature `{feature}` is not supported by this endpoint: {err}"
            );
            Ok(())
        }
        Err(err) => Err(err),
    }
}

fn is_tolerated_live_feature_error(err: &SigmaError) -> bool {
    let SigmaError::ProviderBusiness {
        status,
        code,
        message,
        details,
        ..
    } = err
    else {
        return false;
    };

    if !matches!(
        *status,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY
    ) {
        return false;
    }

    let haystack = format!(
        "{} {} {}",
        code.as_deref().unwrap_or_default(),
        message,
        details
            .as_ref()
            .map_or_else(String::new, ToString::to_string)
    )
    .to_ascii_lowercase();

    [
        "unsupported",
        "not supported",
        "does not support",
        "unknown parameter",
        "unrecognized",
        "invalid parameter",
        "not enabled",
        "not available",
        "beta",
    ]
    .iter()
    .any(|pattern| haystack.contains(pattern))
}

fn assert_provider_business_error(err: SigmaError) {
    match err {
        SigmaError::ProviderBusiness {
            status, message, ..
        } => {
            assert!(!status.is_success());
            assert!(!message.is_empty());
        }
        other => panic!("expected provider business error, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_returns_chat_completion() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };

    let response = client
        .chat()
        .create(&live_request(config.max_tokens))
        .await?;

    assert_text_response(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_stream_yields_chunk() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };

    first_stream_chunk(
        client
            .chat()
            .create_stream(&live_request(config.max_tokens))
            .await?,
    )
    .await?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_stream_collects_text_and_usage() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = full_stream_usage_request(&config);

    let (text, total_tokens) =
        collect_stream_text_and_usage(client.chat().create_stream(&request).await?).await?;

    assert!(
        !text.trim().is_empty(),
        "expected live Anthropic stream to yield content"
    );
    assert!(
        total_tokens.is_some_and(|value| value > 0),
        "expected live Anthropic stream to yield usage"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_returns_usage() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };

    let response = client
        .chat()
        .create(&live_request(config.max_tokens))
        .await?;

    assert_token_usage(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_returns_function_tool_call() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = function_tool_request(&config);

    let response = client.chat().create(&request).await?;

    assert_response_function_tool_call(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_stream_yields_function_tool_call() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = stream_function_tool_request(&config);

    let result =
        collect_stream_function_tool_call(client.chat().create_stream(&request).await?).await?;

    assert_stream_function_tool_call(&result);
    Ok(())
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_hits_prompt_cache_on_repeated_prompt() -> SigmaResult<()> {
    live_feature("prompt_cache", |client, _config| async move {
        let request = prompt_cache_request();

        let warmup = client.chat().create(&request).await?;
        assert!(
            cache_creation_tokens(&warmup) > 0 || cache_read_tokens(&warmup) > 0,
            "expected first live Anthropic cache request to report cache creation or read tokens"
        );

        for attempt in 1..=PROMPT_CACHE_ATTEMPTS {
            let response = client.chat().create(&request).await?;
            let cache_read = cache_read_tokens(&response);
            if cache_read > 0 {
                return Ok(());
            }

            eprintln!(
                "live Anthropic prompt cache attempt {attempt} returned cache_read_input_tokens=0"
            );
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        panic!("expected repeated live Anthropic prompt to report cache_read_input_tokens > 0");
    })
    .await
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_accepts_response_format_json_schema() -> SigmaResult<()> {
    live_feature("response_format_json_schema", |client, config| async move {
        let request = structured_output_request(&config);
        let response = client.chat().create(&request).await?;
        assert_structured_output(&response);
        Ok(())
    })
    .await
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_accepts_provider_options_output_format() -> SigmaResult<()> {
    live_feature(
        "provider_options_output_format",
        |client, config| async move {
            let request = provider_options_output_format_request(&config);
            let response = client.chat().create(&request).await?;
            assert_structured_output(&response);
            Ok(())
        },
    )
    .await
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_create_accepts_reasoning_effort() -> SigmaResult<()> {
    live_feature("reasoning_effort", |client, config| async move {
        let request = reasoning_effort_request(&config);
        let response = client.chat().create(&request).await?;
        assert_chat_response_shape(&response);
        Ok(())
    })
    .await
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and makes a real Anthropic API request"]
async fn live_anthropic_bad_model_maps_to_provider_business_error() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = bad_model_request(&config);

    let err = match client.chat().create(&request).await {
        Ok(response) => panic!("expected bad model error, got {response:?}"),
        Err(err) => err,
    };

    assert_provider_business_error(err);
    Ok(())
}

#[test]
fn live_env_reads_dotenv_values() {
    let env = LiveEnv::from_sources(
        HashMap::new(),
        r#"
        # local Anthropic smoke-test configuration
        export ANTHROPIC_API_KEY='sk-ant-from-dotenv'
        ANTHROPIC_API_BASE="http://localhost:8080"
        ANTHROPIC_MODEL=claude-test-model # inline comment
        "#,
    );

    assert_eq!(
        env.value("ANTHROPIC_API_KEY").as_deref(),
        Some("sk-ant-from-dotenv")
    );
    assert_eq!(
        env.value("ANTHROPIC_API_BASE").as_deref(),
        Some("http://localhost:8080")
    );
    assert_eq!(
        env.value("ANTHROPIC_MODEL").as_deref(),
        Some("claude-test-model")
    );
}

#[test]
fn live_anthropic_config_uses_dotenv_model_configuration() {
    let env = LiveEnv::from_sources(
        HashMap::new(),
        r#"
        ANTHROPIC_API_KEY=sk-ant-from-dotenv
        ANTHROPIC_BASE_URL=http://localhost:8080
        ANTHROPIC_MODEL=claude-dotenv-model
        SIGMA_ANTHROPIC_TEST_BAD_MODEL=claude-impossible-live-test
        SIGMA_ANTHROPIC_TEST_MAX_TOKENS=48
        "#,
    );

    let config = env.anthropic_config().unwrap();

    assert_eq!(config.model, "claude-dotenv-model");
    assert_eq!(config.bad_model, "claude-impossible-live-test");
    assert_eq!(config.api_base.as_deref(), Some("http://localhost:8080"));
    assert_eq!(config.max_tokens, 48);
}

#[test]
fn live_anthropic_config_prefers_process_env_over_dotenv() {
    let env = LiveEnv::from_sources(
        HashMap::from([
            (
                "ANTHROPIC_API_KEY".to_string(),
                "sk-ant-from-env".to_string(),
            ),
            (
                "SIGMA_ANTHROPIC_TEST_MODEL".to_string(),
                "claude-env-model".to_string(),
            ),
        ]),
        r#"
        ANTHROPIC_API_KEY=sk-ant-from-dotenv
        ANTHROPIC_MODEL=claude-dotenv-model
        "#,
    );

    let config = env.anthropic_config().unwrap();

    assert_eq!(config.api_key, "sk-ant-from-env");
    assert_eq!(config.model, "claude-env-model");
}

#[test]
fn live_anthropic_config_defaults_model_and_token_limit() {
    let env = LiveEnv::from_sources(
        HashMap::new(),
        r#"
        ANTHROPIC_API_KEY=sk-ant-from-dotenv
        "#,
    );

    let config = env.anthropic_config().unwrap();

    assert_eq!(config.model, DEFAULT_ANTHROPIC_TEST_MODEL);
    assert_eq!(config.bad_model, DEFAULT_ANTHROPIC_BAD_MODEL);
    assert_eq!(config.max_tokens, DEFAULT_MAX_TOKENS);
}
