use std::collections::HashMap;
use std::sync::Arc;

use crate::config::{ChatParameterMap, ClientConfig};
use crate::provider::{ChatAdapterContext, deployment_model_info};
use crate::types::chat::ChatRequest;
use crate::{
    DeploymentId, ModelDeploymentConfig, ModelName, ModelRef, ProviderDriver, ProviderId,
    SigmaError, SigmaResult,
};

pub(crate) struct DeploymentRouter {
    default_model: Option<ModelName>,
    providers: HashMap<ProviderId, Arc<dyn ProviderDriver>>,
    deployments_by_id: HashMap<DeploymentId, ModelDeploymentConfig>,
    deployments_by_public_model: HashMap<ModelName, DeploymentId>,
}

impl DeploymentRouter {
    pub(crate) fn new(
        config: ClientConfig,
        providers: HashMap<ProviderId, Arc<dyn ProviderDriver>>,
    ) -> SigmaResult<Self> {
        let mut deployments_by_id = HashMap::new();
        let mut deployments_by_public_model = HashMap::new();

        for deployment in &config.deployments {
            if !providers.contains_key(&deployment.provider) {
                return Err(SigmaError::ProviderConfig {
                    provider: Some(deployment.provider.clone()),
                    message: format!(
                        "deployment `{}` references an unknown provider instance",
                        deployment.id
                    ),
                });
            }

            if deployments_by_id
                .insert(deployment.id.clone(), deployment.clone())
                .is_some()
            {
                return Err(SigmaError::DuplicateDeployment {
                    id: deployment.id.clone(),
                });
            }

            if deployments_by_public_model
                .insert(deployment.public_model.clone(), deployment.id.clone())
                .is_some()
            {
                return Err(SigmaError::ProviderConfig {
                    provider: Some(deployment.provider.clone()),
                    message: format!(
                        "multiple deployments expose public model `{}`",
                        deployment.public_model
                    ),
                });
            }
        }

        Ok(Self {
            default_model: config.default_model,
            providers,
            deployments_by_id,
            deployments_by_public_model,
        })
    }

    pub(crate) fn resolve(&self, model: &ModelRef) -> SigmaResult<ResolvedRoute> {
        match model {
            ModelRef::ProviderModel { provider, model } => {
                let provider = self.providers.get(provider).cloned().ok_or_else(|| {
                    SigmaError::ProviderConfig {
                        provider: Some(provider.clone()),
                        message: "unknown provider instance".to_string(),
                    }
                })?;

                Ok(ResolvedRoute {
                    provider,
                    deployment: None,
                    public_model: model.clone(),
                    provider_model: model.clone(),
                })
            }
            ModelRef::Deployment(deployment_id) => {
                let deployment = self
                    .deployments_by_id
                    .get(deployment_id)
                    .cloned()
                    .ok_or_else(|| SigmaError::NoDeploymentForModel {
                        model: model.clone(),
                    })?;
                self.route_for_deployment(deployment)
            }
            ModelRef::Model(model_name) => {
                let model_name = if model_name.as_str().is_empty() {
                    self.default_model.as_ref().ok_or_else(|| {
                        SigmaError::InvalidArgument(
                            "ChatRequest.model is empty and ClientConfig.default_model is not set"
                                .to_string(),
                        )
                    })?
                } else {
                    model_name
                };

                let deployment_id = self
                    .deployments_by_public_model
                    .get(model_name)
                    .ok_or_else(|| SigmaError::NoDeploymentForModel {
                        model: ModelRef::Model(model_name.clone()),
                    })?;
                let deployment = self
                    .deployments_by_id
                    .get(deployment_id)
                    .cloned()
                    .ok_or_else(|| SigmaError::NoDeploymentForModel {
                        model: ModelRef::Model(model_name.clone()),
                    })?;
                self.route_for_deployment(deployment)
            }
        }
    }

    fn route_for_deployment(
        &self,
        deployment: ModelDeploymentConfig,
    ) -> SigmaResult<ResolvedRoute> {
        let provider = self
            .providers
            .get(&deployment.provider)
            .cloned()
            .ok_or_else(|| SigmaError::ProviderConfig {
                provider: Some(deployment.provider.clone()),
                message: format!(
                    "deployment `{}` references an unknown provider instance",
                    deployment.id
                ),
            })?;

        Ok(ResolvedRoute {
            provider,
            public_model: deployment.public_model.clone(),
            provider_model: deployment.provider_model.clone(),
            deployment: Some(deployment),
        })
    }
}

#[derive(Clone)]
pub(crate) struct ResolvedRoute {
    pub(crate) provider: Arc<dyn ProviderDriver>,
    deployment: Option<ModelDeploymentConfig>,
    public_model: ModelName,
    provider_model: ModelName,
}

impl ResolvedRoute {
    pub(crate) fn deployment_defaults(&self) -> Option<&ChatParameterMap> {
        self.deployment
            .as_ref()
            .map(|deployment| &deployment.defaults)
    }

    pub(crate) fn adapter_context(&self) -> ChatAdapterContext<'_> {
        ChatAdapterContext {
            provider: self.provider.id(),
            deployment: self.deployment.as_ref().map(|deployment| &deployment.id),
            public_model: &self.public_model,
            provider_model: &self.provider_model,
            model_info: deployment_model_info(self.deployment.as_ref()),
            provider_state: None,
        }
    }

    pub(crate) fn to_routed_request<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> crate::RoutedChatRequest<'a> {
        crate::RoutedChatRequest {
            provider: self.provider.id(),
            deployment: self.deployment.as_ref().map(|deployment| &deployment.id),
            public_model: &self.public_model,
            provider_model: &self.provider_model,
            model_info: deployment_model_info(self.deployment.as_ref()),
            request,
        }
    }
}
