#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::env;
use std::fs;

use futures_util::StreamExt;
use serde_json::{Value, json};
use sigma::types::chat::{
    ChatMessage, ChatRequest, ChatRequestParams, ChatResponse, FunctionTool,
    NamedFunctionToolChoice, ToolCall, ToolChoice, ToolDefinition, UserMessage,
};
use sigma::types::shared::{FunctionName, FunctionObject};
use sigma::{
    ChatStream, Client, ClientConfig, ModelDeploymentConfig, ModelName, ModelRef,
    ProviderCommonConfig, ProviderConfigMap, ProviderId, ProviderInstanceConfig, ProviderKind,
    SigmaError, SigmaResult,
};

const BEDROCK_PROVIDER_ID: &str = "bedrock-live";
const BEDROCK_PUBLIC_MODEL: &str = "bedrock-live-model";
const BEDROCK_TOOL_PUBLIC_MODEL: &str = "bedrock-live-tool-model";
const DEFAULT_BAD_MODEL: &str = "sigma-nonexistent-bedrock-model";
const DEFAULT_MAX_TOKENS: u32 = 32;
const TOOL_FUNCTION_NAME: &str = "lookup_city_weather";

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiveBedrockConfig {
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    bearer_token: Option<String>,
    region: Option<String>,
    runtime_endpoint: Option<String>,
    model: String,
    tool_model: Option<String>,
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

    fn bedrock_config(&self) -> Option<LiveBedrockConfig> {
        let model = self.first_value(&["SIGMA_BEDROCK_TEST_MODEL", "BEDROCK_MODEL"])?;
        let bearer_token = self.value("AWS_BEARER_TOKEN_BEDROCK");
        let access_key_id = self.value("AWS_ACCESS_KEY_ID");
        let secret_access_key = self.value("AWS_SECRET_ACCESS_KEY");
        if bearer_token.is_none() && !(access_key_id.is_some() && secret_access_key.is_some()) {
            return None;
        }

        Some(LiveBedrockConfig {
            access_key_id,
            secret_access_key,
            session_token: self.value("AWS_SESSION_TOKEN"),
            bearer_token,
            region: self.first_value(&["AWS_REGION_NAME", "AWS_REGION", "AWS_DEFAULT_REGION"]),
            runtime_endpoint: self.value("AWS_BEDROCK_RUNTIME_ENDPOINT"),
            model,
            tool_model: self.value("SIGMA_BEDROCK_TEST_TOOL_MODEL"),
            bad_model: self
                .value("SIGMA_BEDROCK_TEST_BAD_MODEL")
                .unwrap_or_else(|| DEFAULT_BAD_MODEL.to_string()),
            max_tokens: self
                .value("SIGMA_BEDROCK_TEST_MAX_TOKENS")
                .and_then(|value| value.parse::<u32>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_MAX_TOKENS),
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

fn live_setup() -> SigmaResult<Option<(Client, LiveBedrockConfig)>> {
    let Some(config) = LiveEnv::load().bedrock_config() else {
        eprintln!(
            "SIGMA_BEDROCK_TEST_MODEL plus AWS credentials or AWS_BEARER_TOKEN_BEDROCK are not set in the environment or .env; skipping live Bedrock smoke test"
        );
        return Ok(None);
    };

    let client = Client::build(live_client_config(&config))?;
    Ok(Some((client, config)))
}

fn live_tool_setup() -> SigmaResult<Option<(Client, LiveBedrockConfig)>> {
    let Some((client, config)) = live_setup()? else {
        return Ok(None);
    };
    if config.tool_model.is_none() {
        eprintln!(
            "SIGMA_BEDROCK_TEST_TOOL_MODEL is not set in the environment or .env; skipping live Bedrock tool smoke test"
        );
        return Ok(None);
    }

    Ok(Some((client, config)))
}

fn live_client_config(config: &LiveBedrockConfig) -> ClientConfig {
    let mut provider_config = ProviderConfigMap::new();
    insert_optional(&mut provider_config, "access_key_id", &config.access_key_id);
    insert_optional(
        &mut provider_config,
        "secret_access_key",
        &config.secret_access_key,
    );
    insert_optional(&mut provider_config, "session_token", &config.session_token);
    insert_optional(&mut provider_config, "bearer_token", &config.bearer_token);
    insert_optional(&mut provider_config, "region", &config.region);
    insert_optional(
        &mut provider_config,
        "runtime_endpoint",
        &config.runtime_endpoint,
    );

    let mut deployments = vec![ModelDeploymentConfig {
        id: "bedrock-live-chat".into(),
        public_model: ModelName::from(BEDROCK_PUBLIC_MODEL),
        provider: ProviderId::from(BEDROCK_PROVIDER_ID),
        provider_model: ModelName::from(config.model.clone()),
        defaults: serde_json::Map::new(),
        model_info: Value::Null,
    }];
    if let Some(tool_model) = &config.tool_model {
        deployments.push(ModelDeploymentConfig {
            id: "bedrock-live-tool-chat".into(),
            public_model: ModelName::from(BEDROCK_TOOL_PUBLIC_MODEL),
            provider: ProviderId::from(BEDROCK_PROVIDER_ID),
            provider_model: ModelName::from(tool_model.clone()),
            defaults: serde_json::Map::new(),
            model_info: Value::Null,
        });
    }

    ClientConfig {
        providers: vec![ProviderInstanceConfig {
            id: ProviderId::from(BEDROCK_PROVIDER_ID),
            kind: ProviderKind::from("bedrock"),
            common: ProviderCommonConfig {
                api_base: None,
                api_key: None,
                headers: HashMap::new(),
            },
            config: provider_config,
        }],
        deployments,
        default_model: Some(ModelName::from(BEDROCK_PUBLIC_MODEL)),
    }
}

fn insert_optional(map: &mut ProviderConfigMap, key: &str, value: &Option<String>) {
    if let Some(value) = value {
        map.insert(key.to_string(), Value::String(value.clone()));
    }
}

fn live_request(max_tokens: u32) -> ChatRequest {
    request_with(
        ModelRef::model(BEDROCK_PUBLIC_MODEL),
        "Reply with the single word pong.",
        max_tokens,
        |_| {},
    )
}

fn provider_options_request(max_tokens: u32) -> ChatRequest {
    request_with(
        ModelRef::model(BEDROCK_PUBLIC_MODEL),
        "Reply with the single word pong.",
        max_tokens,
        |_| {},
    )
    .with_provider_option(
        ProviderId::from(BEDROCK_PROVIDER_ID),
        "requestMetadata",
        json!({"sigma_live": "bedrock"}),
    )
}

fn bad_model_request(config: &LiveBedrockConfig) -> ChatRequest {
    request_with(
        ModelRef::provider_model(BEDROCK_PROVIDER_ID, config.bad_model.clone()),
        "Reply with pong.",
        config.max_tokens,
        |_| {},
    )
}

fn function_tool_request(config: &LiveBedrockConfig) -> ChatRequest {
    request_with(
        ModelRef::model(BEDROCK_TOOL_PUBLIC_MODEL),
        "Use the weather lookup tool for city San Francisco.",
        config.max_tokens.max(128),
        |params| {
            params.tools = Some(function_tools());
            params.tool_choice = Some(ToolChoice::Function(NamedFunctionToolChoice {
                function: FunctionName {
                    name: TOOL_FUNCTION_NAME.to_string(),
                },
            }));
        },
    )
}

fn request_with(
    model: ModelRef,
    prompt: &str,
    max_tokens: u32,
    update_params: impl FnOnce(&mut ChatRequestParams),
) -> ChatRequest {
    let mut params = ChatRequestParams {
        max_completion_tokens: Some(max_tokens),
        ..Default::default()
    };
    update_params(&mut params);

    ChatRequest::new(model, vec![ChatMessage::User(UserMessage::from(prompt))]).with_params(params)
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
                        "description": "City name"
                    }
                },
                "required": ["city"]
            })),
            strict: None,
        },
    })]
}

fn assert_text_response(response: &ChatResponse) {
    assert!(
        response.choices.iter().any(|choice| {
            choice
                .message
                .content
                .as_deref()
                .is_some_and(|content| !content.trim().is_empty())
        }),
        "expected live Bedrock response to contain assistant text: {response:?}"
    );
}

fn assert_response_function_tool_call(response: &ChatResponse) {
    let tool_calls = response
        .choices
        .first()
        .and_then(|choice| choice.message.tool_calls.as_ref())
        .unwrap_or_else(|| panic!("expected live Bedrock tool call response: {response:?}"));
    let Some(ToolCall::Function(tool_call)) = tool_calls.first() else {
        panic!("expected live Bedrock function tool call: {response:?}");
    };

    assert_eq!(tool_call.function.name, TOOL_FUNCTION_NAME);
    assert!(
        tool_call.function.arguments.contains("San Francisco"),
        "expected tool arguments to mention San Francisco, got {:?}",
        tool_call.function.arguments
    );
}

async fn first_stream_chunk(mut stream: ChatStream) -> SigmaResult<()> {
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.choices.iter().any(|choice| {
            choice
                .delta
                .content
                .as_deref()
                .is_some_and(|content| !content.trim().is_empty())
                || choice
                    .delta
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
                || choice.finish_reason.is_some()
        }) {
            return Ok(());
        }
    }

    panic!("expected live Bedrock stream to yield a semantic chunk");
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
#[ignore = "requires AWS Bedrock credentials, SIGMA_BEDROCK_TEST_MODEL, and makes a real Bedrock API request"]
async fn live_bedrock_create_returns_chat_completion() -> SigmaResult<()> {
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
#[ignore = "requires AWS Bedrock credentials, SIGMA_BEDROCK_TEST_MODEL, and makes a real Bedrock API request"]
async fn live_bedrock_create_stream_yields_chunk() -> SigmaResult<()> {
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
#[ignore = "requires AWS Bedrock credentials, SIGMA_BEDROCK_TEST_MODEL, and makes a real Bedrock API request"]
async fn live_bedrock_create_accepts_request_metadata_provider_option() -> SigmaResult<()> {
    let Some((client, config)) = live_setup()? else {
        return Ok(());
    };

    let response = client
        .chat()
        .create(&provider_options_request(config.max_tokens))
        .await?;

    assert_text_response(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires AWS Bedrock credentials, SIGMA_BEDROCK_TEST_TOOL_MODEL, and makes a real Bedrock API request"]
async fn live_bedrock_create_returns_function_tool_call() -> SigmaResult<()> {
    let Some((client, config)) = live_tool_setup()? else {
        return Ok(());
    };

    let response = client
        .chat()
        .create(&function_tool_request(&config))
        .await?;

    assert_response_function_tool_call(&response);
    Ok(())
}

#[tokio::test]
#[ignore = "requires AWS Bedrock credentials, SIGMA_BEDROCK_TEST_MODEL, and makes a real Bedrock API request"]
async fn live_bedrock_bad_model_maps_to_provider_business_error() -> SigmaResult<()> {
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
        # local Bedrock smoke-test configuration
        export AWS_ACCESS_KEY_ID='AKIA_DOTENV'
        AWS_SECRET_ACCESS_KEY="secret-from-dotenv"
        AWS_REGION=us-east-1 # inline comment
        SIGMA_BEDROCK_TEST_MODEL=anthropic.claude-test
        "#,
    );

    assert_eq!(
        env.value("AWS_ACCESS_KEY_ID").as_deref(),
        Some("AKIA_DOTENV")
    );
    assert_eq!(
        env.value("AWS_SECRET_ACCESS_KEY").as_deref(),
        Some("secret-from-dotenv")
    );
    assert_eq!(env.value("AWS_REGION").as_deref(), Some("us-east-1"));
    assert_eq!(
        env.value("SIGMA_BEDROCK_TEST_MODEL").as_deref(),
        Some("anthropic.claude-test")
    );
}

#[test]
fn live_bedrock_config_uses_dotenv_model_configuration() {
    let env = LiveEnv::from_sources(
        HashMap::new(),
        r#"
        AWS_BEARER_TOKEN_BEDROCK=bedrock-token-from-dotenv
        AWS_REGION_NAME=us-west-2
        AWS_BEDROCK_RUNTIME_ENDPOINT=http://localhost:8080
        SIGMA_BEDROCK_TEST_MODEL=bedrock-dotenv-model
        SIGMA_BEDROCK_TEST_TOOL_MODEL=bedrock-tool-dotenv-model
        SIGMA_BEDROCK_TEST_BAD_MODEL=bedrock-impossible-live-test
        SIGMA_BEDROCK_TEST_MAX_TOKENS=48
        "#,
    );

    let config = env.bedrock_config().unwrap();

    assert_eq!(
        config.bearer_token.as_deref(),
        Some("bedrock-token-from-dotenv")
    );
    assert_eq!(config.region.as_deref(), Some("us-west-2"));
    assert_eq!(
        config.runtime_endpoint.as_deref(),
        Some("http://localhost:8080")
    );
    assert_eq!(config.model, "bedrock-dotenv-model");
    assert_eq!(
        config.tool_model.as_deref(),
        Some("bedrock-tool-dotenv-model")
    );
    assert_eq!(config.bad_model, "bedrock-impossible-live-test");
    assert_eq!(config.max_tokens, 48);
}

#[test]
fn live_bedrock_config_prefers_process_env_over_dotenv() {
    let env = LiveEnv::from_sources(
        HashMap::from([
            ("AWS_ACCESS_KEY_ID".to_string(), "AKIA_ENV".to_string()),
            (
                "AWS_SECRET_ACCESS_KEY".to_string(),
                "secret-from-env".to_string(),
            ),
            (
                "SIGMA_BEDROCK_TEST_MODEL".to_string(),
                "bedrock-env-model".to_string(),
            ),
        ]),
        r#"
        AWS_ACCESS_KEY_ID=AKIA_DOTENV
        AWS_SECRET_ACCESS_KEY=secret-from-dotenv
        SIGMA_BEDROCK_TEST_MODEL=bedrock-dotenv-model
        "#,
    );

    let config = env.bedrock_config().unwrap();

    assert_eq!(config.access_key_id.as_deref(), Some("AKIA_ENV"));
    assert_eq!(config.secret_access_key.as_deref(), Some("secret-from-env"));
    assert_eq!(config.model, "bedrock-env-model");
}

#[test]
fn live_bedrock_config_requires_model_and_auth() {
    let env = LiveEnv::from_sources(
        HashMap::new(),
        r#"
        AWS_REGION=us-east-1
        SIGMA_BEDROCK_TEST_MODEL=bedrock-dotenv-model
        "#,
    );
    assert!(env.bedrock_config().is_none());

    let env = LiveEnv::from_sources(
        HashMap::new(),
        r#"
        AWS_ACCESS_KEY_ID=AKIA_DOTENV
        AWS_SECRET_ACCESS_KEY=secret-from-dotenv
        "#,
    );
    assert!(env.bedrock_config().is_none());
}
