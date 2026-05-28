use std::collections::HashMap;
use std::sync::Arc;

use futures_util::StreamExt;

use crate::config::ClientConfig;
use crate::provider::{
    ChatAdapterContext, ChatAdapterRequest, ChatStream, ProviderCatalog, ProviderInit,
};
use crate::route::{DeploymentRouter, ResolvedRoute};
use crate::types::chat::{ChatRequest, ChatResponse};
use crate::{SigmaError, SigmaResult};

/// Entry point for calling configured LLM providers.
///
/// A client owns initialized provider instances, deployment routing tables, and
/// a shared [`reqwest::Client`]. Build one from [`ClientConfig`] and reuse it
/// across requests.
///
/// ```rust
/// # use sigma::{Client, ClientConfig};
/// # fn example() -> sigma::SigmaResult<()> {
/// let client = Client::build(ClientConfig::default())?;
/// # let request = sigma::types::chat::ChatRequest::new(
/// #     sigma::ModelRef::model("gpt-4o"),
/// #     Vec::<sigma::types::chat::ChatMessage>::new(),
/// # );
/// # let _future = client.create(&request);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    routes: DeploymentRouter,
    http_client: reqwest::Client,
}

/// Builder for runtime resources shared by a [`Client`].
///
/// Provider constructors come from the inventory registry. This builder only
/// accepts runtime resources, such as the HTTP client used for provider calls.
#[derive(Clone)]
pub struct ClientBuilder {
    http_client: reqwest::Client,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self {
            http_client: reqwest::Client::new(),
        }
    }
}

impl ClientBuilder {
    /// Installs the reqwest client used for provider HTTP calls.
    ///
    /// The same client is used for all provider instances. Applications can
    /// configure the [`reqwest::Client`] with their own timeouts, proxies, TLS,
    /// tracing middleware, or test server settings before passing it here.
    pub fn with_http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = http_client;
        self
    }

    /// Builds a client from runtime resources and [`ClientConfig`].
    ///
    /// # Errors
    ///
    /// Returns an error when provider registrations are duplicated, a provider
    /// kind is unknown, provider instance ids are duplicated, deployments are
    /// duplicated, or a deployment references an unknown provider instance.
    pub fn build(self, config: ClientConfig) -> SigmaResult<Client> {
        let catalog = ProviderCatalog::from_inventory()?;
        let mut providers = HashMap::new();

        for provider_config in &config.providers {
            if providers.contains_key(&provider_config.id) {
                return Err(SigmaError::DuplicateProviderInstance {
                    id: provider_config.id.clone(),
                });
            }

            let constructor = catalog.get(&provider_config.kind).ok_or_else(|| {
                SigmaError::UnknownProviderKind {
                    kind: provider_config.kind.clone(),
                }
            })?;
            let provider = constructor(ProviderInit::from(provider_config.clone()))?;
            providers.insert(provider_config.id.clone(), provider);
        }

        let routes = DeploymentRouter::new(config, providers)?;

        Ok(Client {
            inner: Arc::new(ClientInner {
                routes,
                http_client: self.http_client,
            }),
        })
    }
}

impl Client {
    /// Starts a client builder.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Builds a client with default runtime resources.
    ///
    /// This uses a default [`reqwest::Client`]. Applications that need custom
    /// HTTP runtime behavior should call [`Client::builder`] and configure it
    /// with [`ClientBuilder::with_http_client`].
    ///
    /// # Errors
    ///
    /// Returns the same configuration errors as [`ClientBuilder::build`].
    pub fn build(config: ClientConfig) -> SigmaResult<Self> {
        Self::builder().build(config)
    }

    /// Creates one chat completion.
    ///
    /// The call resolves [`ChatRequest::model`] through deployment routing,
    /// runs the provider adapter lifecycle, sends the signed request with the
    /// configured HTTP client, and transforms the provider response back into
    /// sigma's semantic chat response type.
    ///
    /// # Errors
    ///
    /// Returns routing, unsupported parameter, provider adapter, HTTP, or
    /// provider response errors.
    pub async fn create(&self, request: &ChatRequest) -> SigmaResult<ChatResponse> {
        self.create_chat_completion(request).await
    }

    /// Creates a streaming chat completion.
    ///
    /// The call resolves [`ChatRequest::model`] through deployment routing,
    /// prepares the provider request with streaming enabled, sends it through
    /// the shared HTTP streaming path, and forwards provider bytes to the
    /// adapter's stream decoder.
    ///
    /// # Errors
    ///
    /// Returns routing, unsupported parameter, provider adapter, HTTP, or
    /// provider response errors.
    pub async fn create_stream(&self, request: &ChatRequest) -> SigmaResult<ChatStream> {
        self.create_chat_completion_stream(request).await
    }
}

impl Client {
    async fn create_chat_completion(
        &self,
        request: &ChatRequest,
    ) -> SigmaResult<crate::types::chat::ChatResponse> {
        let route = self.inner.routes.resolve(&request.model)?;

        if let Some(custom_chat) = route.provider.custom_chat() {
            return custom_chat.create(route.to_routed_request(request)).await;
        }

        let provider = Arc::clone(&route.provider);
        let adapter = provider
            .chat()
            .ok_or_else(|| SigmaError::ProviderDoesNotSupportChat {
                provider: provider.id().clone(),
            })?;
        let (signed_request, adapter, context) =
            self.prepare_provider_request(request, &route, adapter, false)?;
        let response = self.execute_http(signed_request).await?;

        transform_response_or_error(adapter, &context, response)
    }

    async fn create_chat_completion_stream(
        &self,
        request: &ChatRequest,
    ) -> SigmaResult<ChatStream> {
        let route = self.inner.routes.resolve(&request.model)?;

        if let Some(custom_chat) = route.provider.custom_chat() {
            return custom_chat
                .create_stream(route.to_routed_request(request))
                .await;
        }

        let provider = Arc::clone(&route.provider);
        let adapter = provider
            .chat()
            .ok_or_else(|| SigmaError::ProviderDoesNotSupportChat {
                provider: provider.id().clone(),
            })?;
        let (signed_request, adapter, context) =
            self.prepare_provider_request(request, &route, adapter, true)?;

        let byte_stream = self.stream_http(signed_request, adapter, &context).await?;
        adapter.transform_stream(&context, byte_stream)
    }

    async fn execute_http(
        &self,
        request: crate::SignedProviderRequest,
    ) -> SigmaResult<crate::ProviderResponse> {
        let response = self
            .http_request(request)
            .send()
            .await
            .map_err(http_error)?;

        provider_response(response).await
    }

    async fn stream_http(
        &self,
        request: crate::SignedProviderRequest,
        adapter: &dyn crate::ChatCompletionAdapter,
        context: &crate::ChatAdapterContext<'_>,
    ) -> SigmaResult<crate::ProviderByteStream> {
        let response = self
            .http_request(request)
            .send()
            .await
            .map_err(http_error)?;
        let status = response.status();

        if !status.is_success() {
            let response = provider_response(response).await?;
            return Err(adapter.transform_error_response(context, response));
        }

        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(http_error));

        Ok(Box::pin(stream))
    }

    fn http_request(&self, request: crate::SignedProviderRequest) -> reqwest::RequestBuilder {
        self.inner
            .http_client
            .request(request.method, request.url)
            .headers(request.headers)
            .body(request.body.to_string())
    }

    fn prepare_provider_request<'a>(
        &self,
        request: &'a ChatRequest,
        route: &'a ResolvedRoute,
        adapter: &'a dyn crate::ChatCompletionAdapter,
        streaming: bool,
    ) -> SigmaResult<(
        crate::SignedProviderRequest,
        &'a dyn crate::ChatCompletionAdapter,
        crate::ChatAdapterContext<'a>,
    )> {
        let context = route.adapter_context();
        let adapter_request = ChatAdapterRequest {
            context: context.clone(),
            request,
            deployment_defaults: route.deployment_defaults(),
            streaming,
        };

        let endpoint = adapter.endpoint(&adapter_request)?;
        let provider_request = adapter.transform_request(adapter_request, endpoint)?;
        let signed_request = adapter.sign_request(provider_request)?;
        let context = ChatAdapterContext {
            provider_state: signed_request.provider_state.clone(),
            ..context
        };

        Ok((signed_request, adapter, context))
    }
}

fn http_error(error: reqwest::Error) -> SigmaError {
    SigmaError::Http {
        message: error.to_string(),
    }
}

fn transform_response_or_error(
    adapter: &dyn crate::ChatCompletionAdapter,
    context: &crate::ChatAdapterContext<'_>,
    response: crate::ProviderResponse,
) -> SigmaResult<ChatResponse> {
    if response.status.is_success() {
        adapter.transform_response(context, response)
    } else {
        Err(adapter.transform_error_response(context, response))
    }
}

async fn provider_response(response: reqwest::Response) -> SigmaResult<crate::ProviderResponse> {
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.bytes().await.map_err(http_error)?;

    Ok(crate::ProviderResponse {
        status,
        headers,
        body,
    })
}
