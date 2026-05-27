# Vendored model capability data

`models.dev.json` is a snapshot of [anomalyco/models.dev](https://github.com/anomalyco/models.dev)
(MIT licensed) used by `sigma::model_capabilities` to look up per-model
capabilities such as context window, thinking mode, and supported sampling
parameters.

The snapshot is a verbatim copy of the upstream `models.json` from the `dev`
branch. Refresh by running:

```sh
curl -L https://raw.githubusercontent.com/anomalyco/models.dev/dev/models.json \
  -o data/models.dev.json
```

Callers can override sigma's defaults per deployment by populating
`ModelDeploymentConfig::model_info` with a JSON object whose fields match the
public `ModelCapabilities` schema.
