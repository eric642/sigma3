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
    ChatMessage, ChatRequest, ChatRequestParams, ChatResponse, DeveloperMessage, FunctionTool,
    NamedFunctionToolChoice, PredictionContent, PredictionContentValue, PromptCacheRetention,
    ServiceTier, StreamOptions, TextContent, TextPart, ToolCall, ToolChoice, ToolDefinition,
    UserContent, UserContentPart, UserMessage,
};
use sigma::types::shared::{FunctionName, FunctionObject, ResponseFormat};
use sigma::{
    ChatStream, Client, ClientConfig, ModelDeploymentConfig, ModelName, ModelRef,
    ProviderCommonConfig, ProviderConfigMap, ProviderId, ProviderInstanceConfig, ProviderKind,
    SecretString, SigmaError, SigmaResult,
};

const DEFAULT_OPENAI_TEST_MODEL: &str = "gpt-4o-mini";
const DEFAULT_OPENAI_BAD_MODEL: &str = "sigma-nonexistent-openai-model";
const DEFAULT_MAX_COMPLETION_TOKENS: u32 = 16;
const PROMPT_CACHE_ATTEMPTS: u8 = 3;
const PROMPT_CACHE_KEY: &str = "sigma-live-openai-cache";
const TOOL_PROMPT_CACHE_KEY: &str = "sigma-live-openai-tool-cache";
const TOOL_FUNCTION_NAME: &str = "lookup_city_weather";

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiveOpenAiConfig {
    api_key: String,
    api_base: Option<String>,
    model: String,
    bad_model: String,
    max_completion_tokens: u32,
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

    fn openai_config(&self) -> Option<LiveOpenAiConfig> {
        let api_key = self.value("OPENAI_API_KEY")?;
        let api_base = self.first_value(&["OPENAI_BASE_URL", "OPENAI_API_BASE"]);
        let model = self
            .first_value(&["SIGMA_OPENAI_TEST_MODEL", "OPENAI_MODEL"])
            .unwrap_or_else(|| DEFAULT_OPENAI_TEST_MODEL.to_string());
        let bad_model = self
            .value("SIGMA_OPENAI_TEST_BAD_MODEL")
            .unwrap_or_else(|| DEFAULT_OPENAI_BAD_MODEL.to_string());
        let max_completion_tokens = self
            .value("SIGMA_OPENAI_TEST_MAX_COMPLETION_TOKENS")
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS);

        Some(LiveOpenAiConfig {
            api_key,
            api_base,
            model,
            bad_model,
            max_completion_tokens,
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

fn live_client_config(config: &LiveOpenAiConfig) -> ClientConfig {
    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: ProviderId::from("openai-live"),
            kind: ProviderKind::from("openai"),
            common: ProviderCommonConfig {
                api_base: config.api_base.clone(),
                api_key: Some(SecretString::from(config.api_key.clone())),
                headers: HashMap::new(),
            },
            config: ProviderConfigMap::new(),
        }],
        deployments: vec![ModelDeploymentConfig {
            id: "openai-live-chat".into(),
            public_model: ModelName::from("openai-live-model"),
            provider: ProviderId::from("openai-live"),
            provider_model: ModelName::from(config.model.clone()),
            defaults: serde_json::Map::new(),
            model_info: serde_json::Value::Null,
        }],
        default_model: Some(ModelName::from("openai-live-model")),
    }
}

fn live_request(max_completion_tokens: u32) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message("Reply with the single word pong.")],
        ModelRef::model("openai-live-model"),
        max_completion_tokens,
        |_| {},
    )
}

fn live_setup() -> SigmaResult<Option<(Client, LiveOpenAiConfig)>> {
    let Some(config) = LiveEnv::load().openai_config() else {
        eprintln!(
            "OPENAI_API_KEY is not set in the environment or .env; skipping live OpenAI smoke test"
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
    max_completion_tokens: u32,
    update_params: impl FnOnce(&mut ChatRequestParams),
) -> SigmaResult<ChatRequest> {
    let mut params = ChatRequestParams {
        max_completion_tokens: Some(max_completion_tokens),
        ..Default::default()
    };
    update_params(&mut params);

    Ok(ChatRequest::new(model, messages).with_params(params))
}

fn content_part_request(content: &str, config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![ChatMessage::User(UserMessage {
            content: UserContent::Parts(vec![UserContentPart::Text(TextPart {
                text: content.to_string(),
                cache_control: None,
            })]),
            name: None,
        })],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens,
        |_| {},
    )
}

fn named_user_request(name: &str, config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![ChatMessage::User(UserMessage {
            content: UserContent::Text("Reply briefly.".to_string()),
            name: Some(name.to_string()),
        })],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens,
        |_| {},
    )
}

fn developer_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![
            ChatMessage::Developer(DeveloperMessage {
                content: TextContent::Text("Keep the answer short.".to_string()),
                name: None,
            }),
            user_text_message("Reply with pong."),
        ],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens,
        |_| {},
    )
}

fn stream_options_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message("Reply with pong.")],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens,
        |params| {
            params.stream_options = Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
        },
    )
}

fn full_stream_usage_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message("Reply exactly with sigma-stream-pong.")],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens.max(8),
        |params| {
            params.stream_options = Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
        },
    )
}

fn token_logprobs_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message("Reply with exactly one short word.")],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens.max(4),
        |params| {
            params.logprobs = Some(true);
            params.top_logprobs = Some(1);
        },
    )
}

fn prompt_cache_request() -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message(&prompt_cache_prompt())],
        ModelRef::model("openai-live-model"),
        DEFAULT_MAX_COMPLETION_TOKENS,
        |params| {
            params.prompt_cache_key = Some(PROMPT_CACHE_KEY.to_string());
            params.prompt_cache_retention = Some(PromptCacheRetention::TwentyFourHours);
        },
    )
}

fn tool_prompt_cache_request() -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message(&tool_prompt_cache_prompt())],
        ModelRef::model("openai-live-model"),
        64,
        |params| {
            configure_function_tool(params);
            params.prompt_cache_key = Some(TOOL_PROMPT_CACHE_KEY.to_string());
            params.prompt_cache_retention = Some(PromptCacheRetention::TwentyFourHours);
        },
    )
}

fn prompt_cache_prompt() -> String {
    let stable_prefix = (0..1600)
        .map(|index| format!("cache-prefix-{index:04}"))
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        "{stable_prefix}\n\nThe preceding prefix is intentionally stable for prompt cache testing. Reply with ok."
    )
}

fn tool_prompt_cache_prompt() -> String {
    let stable_prefix = (0..1600)
        .map(|index| format!("tool-cache-prefix-{index:04}"))
        .collect::<Vec<_>>()
        .join(" ");

    format!("{stable_prefix}\n\nUse the {TOOL_FUNCTION_NAME} tool for city San Francisco.")
}

fn json_object_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message(
            "Return a JSON object with exactly one boolean field named ok.",
        )],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens.max(24),
        |params| {
            params.response_format = Some(ResponseFormat::JsonObject);
        },
    )
}

fn function_tool_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message(
            "Use the weather lookup tool for city San Francisco.",
        )],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens.max(64),
        configure_function_tool,
    )
}

fn stream_function_tool_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message(
            "Use the weather lookup tool for city San Francisco.",
        )],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens.max(64),
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

fn safety_identifier_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message("Reply with pong.")],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens,
        |params| {
            params.safety_identifier = Some("sigma-live-test-user".to_string());
        },
    )
}

fn prediction_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message("Return exactly: pong")],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens,
        |params| {
            params.prediction = Some(PredictionContent::Content(PredictionContentValue::Text(
                "pong".to_string(),
            )));
        },
    )
}

fn service_tier_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message("Reply with pong.")],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens,
        |params| {
            params.service_tier = Some(ServiceTier::Priority);
        },
    )
}

fn stream_n_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message("Say hello in one word.")],
        ModelRef::model("openai-live-model"),
        config.max_completion_tokens,
        |params| {
            params.count = Some(2);
        },
    )
}

fn bad_model_request(config: &LiveOpenAiConfig) -> SigmaResult<ChatRequest> {
    request_with(
        vec![user_text_message(
            "This request should fail because the model is invalid.",
        )],
        ModelRef::provider_model(ProviderId::from("openai-live"), config.bad_model.clone()),
        config.max_completion_tokens,
        |_| {},
    )
}

fn assert_chat_response_shape(response: &ChatResponse) {
    assert!(
        !response.id.is_empty() || !response.model.is_empty() || !response.object.is_empty(),
        "expected at least one identifying response field"
    );
    assert!(
        !response.choices.is_empty(),
        "expected at least one chat completion choice"
    );
}

async fn first_stream_chunk(mut stream: ChatStream) -> SigmaResult<()> {
    let Some(chunk) = stream.next().await else {
        panic!("expected live OpenAI stream to yield at least one chunk");
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

    assert!(saw_chunk, "expected live OpenAI stream to yield chunks");
    Ok((text, total_tokens))
}

fn assert_token_usage(response: &ChatResponse) {
    let usage = response
        .usage
        .as_ref()
        .unwrap_or_else(|| panic!("expected live OpenAI response usage"));

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

fn assert_logprobs(response: &ChatResponse) {
    let content = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|logprobs| logprobs.content.as_ref())
        .unwrap_or_else(|| panic!("expected live OpenAI token logprobs"));
    let first = content
        .first()
        .unwrap_or_else(|| panic!("expected at least one token logprob"));

    assert!(!first.token.is_empty(), "expected non-empty token text");
    assert!(first.logprob.is_finite(), "expected finite token logprob");
}

fn cached_tokens(response: &ChatResponse) -> u32 {
    response
        .usage
        .as_ref()
        .and_then(|usage| usage.prompt_tokens_details.as_ref())
        .and_then(|details| details.cached_tokens)
        .unwrap_or(0)
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

                if let Some(id) = tool_call.id.filter(|id| !id.is_empty()) {
                    result.id.get_or_insert(id);
                }

                let Some(function) = tool_call.function else {
                    continue;
                };
                if let Some(name) = function.name.filter(|name| !name.is_empty()) {
                    result.name.get_or_insert(name);
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
    assert_token_usage(response);

    let tool_calls = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap_or_else(|| panic!("expected live OpenAI response tool_calls"));
    let tool_call = tool_calls
        .first()
        .unwrap_or_else(|| panic!("expected at least one live OpenAI tool call"));

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

async fn live_feature<F, Fut>(feature: &'static str, f: F) -> SigmaResult<()>
where
    F: FnOnce(Client, LiveOpenAiConfig) -> Fut,
    Fut: Future<Output = SigmaResult<()>>,
{
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };

    match f(client, config).await {
        Ok(()) => Ok(()),
        Err(err) if is_tolerated_live_feature_error(&err) => {
            eprintln!("live OpenAI feature `{feature}` is not supported by this endpoint: {err}");
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
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_returns_chat_completion() -> SigmaResult<()> {
    let Some(config) = LiveEnv::load().openai_config() else {
        eprintln!(
            "OPENAI_API_KEY is not set in the environment or .env; skipping live OpenAI smoke test"
        );
        return Ok(());
    };
    let client = Client::build(live_client_config(&config))?;

    let response = client
        .create(&live_request(config.max_completion_tokens)?)
        .await?;

    assert_eq!(response.object, "chat.completion");
    assert!(!response.id.is_empty());
    assert!(!response.model.is_empty());
    assert!(!response.choices.is_empty());
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_stream_yields_chunk() -> SigmaResult<()> {
    let Some(config) = LiveEnv::load().openai_config() else {
        eprintln!(
            "OPENAI_API_KEY is not set in the environment or .env; skipping live OpenAI smoke test"
        );
        return Ok(());
    };
    let client = Client::build(live_client_config(&config))?;

    let mut stream = client
        .create_stream(&live_request(config.max_completion_tokens)?)
        .await?;
    let Some(chunk) = stream.next().await else {
        panic!("expected live OpenAI stream to yield at least one chunk");
    };
    let _chunk = chunk?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_content_part_array() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = content_part_request("Hello from a text content part.", &config)?;

    let response = client.create(&request).await?;

    assert_chat_response_shape(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_user_message_name() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = named_user_request("test_user", &config)?;

    let response = client.create(&request).await?;

    assert_chat_response_shape(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_developer_message() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = developer_request(&config)?;

    let response = client.create(&request).await?;

    assert_chat_response_shape(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_stream_with_include_usage_yields_chunk() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = stream_options_request(&config)?;

    first_stream_chunk(client.create_stream(&request).await?).await?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_stream_collects_text_and_usage() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = full_stream_usage_request(&config)?;

    let (text, total_tokens) =
        collect_stream_text_and_usage(client.create_stream(&request).await?).await?;

    assert!(
        !text.trim().is_empty(),
        "expected live OpenAI stream to yield content"
    );
    assert!(
        total_tokens.is_some_and(|value| value > 0),
        "expected live OpenAI stream to yield usage"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_returns_usage() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = live_request(config.max_completion_tokens)?;

    let response = client.create(&request).await?;

    assert_token_usage(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_logprobs() -> SigmaResult<()> {
    live_feature("logprobs", |client, config| async move {
        let request = token_logprobs_request(&config)?;
        let response = client.create(&request).await?;
        assert_token_usage(&response);
        assert_logprobs(&response);
        Ok(())
    })
    .await
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_hits_prompt_cache_on_repeated_prompt() -> SigmaResult<()> {
    let Some((client, _config)) = live_setup()? else {
        return Ok(());
    };
    let request = prompt_cache_request()?;

    let warmup = client.create(&request).await?;
    assert_token_usage(&warmup);

    for attempt in 1..=PROMPT_CACHE_ATTEMPTS {
        let response = client.create(&request).await?;
        let cached_tokens = cached_tokens(&response);
        if cached_tokens > 0 {
            return Ok(());
        }

        eprintln!("live OpenAI prompt cache attempt {attempt} returned cached_tokens=0");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    panic!(
        "expected repeated live OpenAI prompt to report prompt_tokens_details.cached_tokens > 0"
    );
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_hits_prompt_cache_with_tool() -> SigmaResult<()> {
    let Some((client, _config)) = live_setup()? else {
        return Ok(());
    };
    let request = tool_prompt_cache_request()?;

    let warmup = client.create(&request).await?;
    assert_response_function_tool_call(&warmup);

    for attempt in 1..=PROMPT_CACHE_ATTEMPTS {
        let response = client.create(&request).await?;
        assert_response_function_tool_call(&response);

        let cached_tokens = cached_tokens(&response);
        if cached_tokens > 0 {
            return Ok(());
        }

        eprintln!("live OpenAI tool prompt cache attempt {attempt} returned cached_tokens=0");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    panic!(
        "expected repeated live OpenAI tool prompt to report prompt_tokens_details.cached_tokens > 0"
    );
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_response_format_json_object() -> SigmaResult<()> {
    live_feature("response_format_json_object", |client, config| async move {
        let request = json_object_request(&config)?;
        let response = client.create(&request).await?;
        assert_chat_response_shape(&response);
        Ok(())
    })
    .await
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_function_tool() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = function_tool_request(&config)?;

    let response = client.create(&request).await?;

    assert_response_function_tool_call(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_stream_yields_function_tool_call() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = stream_function_tool_request(&config)?;

    let result = collect_stream_function_tool_call(client.create_stream(&request).await?).await?;

    assert_stream_function_tool_call(&result);
    Ok(())
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_safety_identifier() -> SigmaResult<()> {
    live_feature("safety_identifier", |client, config| async move {
        let request = safety_identifier_request(&config)?;
        let response = client.create(&request).await?;
        assert_chat_response_shape(&response);
        Ok(())
    })
    .await
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_prediction() -> SigmaResult<()> {
    live_feature("prediction", |client, config| async move {
        let request = prediction_request(&config)?;
        let response = client.create(&request).await?;
        assert_chat_response_shape(&response);
        Ok(())
    })
    .await
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_create_accepts_service_tier() -> SigmaResult<()> {
    live_feature("service_tier", |client, config| async move {
        let request = service_tier_request(&config)?;
        let response = client.create(&request).await?;
        assert_chat_response_shape(&response);
        Ok(())
    })
    .await
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_stream_accepts_n_greater_than_one() -> SigmaResult<()> {
    live_feature("stream_n_greater_than_one", |client, config| async move {
        let request = stream_n_request(&config)?;
        first_stream_chunk(client.create_stream(&request).await?).await?;
        Ok(())
    })
    .await
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and makes a real OpenAI API request"]
async fn live_openai_bad_model_maps_to_provider_business_error() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };
    let request = bad_model_request(&config)?;

    let err = match client.create(&request).await {
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
        # local OpenAI smoke-test configuration
        export OPENAI_API_KEY='sk-from-dotenv'
        OPENAI_BASE_URL="http://localhost:8080/v1"
        OPENAI_MODEL=gpt-4.1-mini # inline comment
        "#,
    );

    assert_eq!(
        env.value("OPENAI_API_KEY").as_deref(),
        Some("sk-from-dotenv")
    );
    assert_eq!(
        env.value("OPENAI_BASE_URL").as_deref(),
        Some("http://localhost:8080/v1")
    );
    assert_eq!(env.value("OPENAI_MODEL").as_deref(), Some("gpt-4.1-mini"));
}

#[test]
fn live_openai_config_uses_dotenv_model_configuration() {
    let env = LiveEnv::from_sources(
        HashMap::new(),
        r#"
        OPENAI_API_KEY=sk-from-dotenv
        OPENAI_BASE_URL=http://localhost:8080/v1
        OPENAI_MODEL=gpt-4.1-mini
        SIGMA_OPENAI_TEST_BAD_MODEL=gpt-impossible-live-test
        SIGMA_OPENAI_TEST_MAX_COMPLETION_TOKENS=24
        "#,
    );

    let config = env.openai_config().unwrap();

    assert_eq!(config.model, "gpt-4.1-mini");
    assert_eq!(config.bad_model, "gpt-impossible-live-test");
    assert_eq!(config.api_base.as_deref(), Some("http://localhost:8080/v1"));
    assert_eq!(config.max_completion_tokens, 24);
}

#[test]
fn live_openai_config_prefers_process_env_over_dotenv() {
    let env = LiveEnv::from_sources(
        HashMap::from([
            ("OPENAI_API_KEY".to_string(), "sk-from-env".to_string()),
            (
                "SIGMA_OPENAI_TEST_MODEL".to_string(),
                "gpt-env-model".to_string(),
            ),
        ]),
        r#"
        OPENAI_API_KEY=sk-from-dotenv
        OPENAI_MODEL=gpt-dotenv-model
        "#,
    );

    let config = env.openai_config().unwrap();

    assert_eq!(config.api_key, "sk-from-env");
    assert_eq!(config.model, "gpt-env-model");
}

#[test]
fn live_openai_config_defaults_model_and_token_limit() {
    let env = LiveEnv::from_sources(
        HashMap::new(),
        r#"
        OPENAI_API_KEY=sk-from-dotenv
        "#,
    );

    let config = env.openai_config().unwrap();

    assert_eq!(config.model, "gpt-4o-mini");
    assert_eq!(config.bad_model, DEFAULT_OPENAI_BAD_MODEL);
    assert_eq!(config.max_completion_tokens, 16);
}
