use crate::{
    config::{LlmConfig, Provider},
    error::LlmError,
    traits::LlmClient,
};

/// Controls the expected output format.
#[derive(Clone)]
pub enum ResponseShape {
    /// Plain text response. Default.
    Text,
    /// JSON response. Providers that support schema enforcement use it natively;
    /// Ollama receives the schema serialised into the instruction instead.
    Json(serde_json::Value),
}

/// Internal configuration passed from builder to provider constructors.
pub(crate) struct ClientConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model: String,
    pub instruction: String,
    pub max_tokens: u32,
    pub thinking: bool,
    pub response_shape: ResponseShape,
}

/// Fluent builder for LLM clients. Connectivity comes from `LlmConfig` or
/// individual setter calls. Behavioural fields are application-level and never
/// stored in `LlmConfig`.
#[derive(Default)]
pub struct LlmClientBuilder {
    // connectivity
    provider: Option<Provider>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,

    // behaviour
    instruction: Option<String>,
    max_tokens: Option<u32>,
    thinking: Option<bool>,
    response_shape: Option<ResponseShape>,
}

impl LlmClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets provider, api_key, base_url, and model from an `LlmConfig` at once.
    pub fn config(self, c: LlmConfig) -> Self {
        Self {
            provider: Some(c.provider),
            api_key: Some(c.api_key),
            base_url: c.base_url,
            model: Some(c.model),
            ..self
        }
    }

    /// Sets the provider. Required if [`config`](LlmClientBuilder::config) is not used.
    pub fn provider(self, p: Provider) -> Self {
        Self {
            provider: Some(p),
            ..self
        }
    }

    /// Sets the API key. Required for all providers except Ollama.
    pub fn api_key(self, k: impl Into<String>) -> Self {
        Self {
            api_key: Some(k.into()),
            ..self
        }
    }

    /// Sets the base URL. Required for Ollama; ignored by hosted providers.
    pub fn base_url(self, u: impl Into<String>) -> Self {
        Self {
            base_url: Some(u.into()),
            ..self
        }
    }

    /// Overrides the model. Falls back to [`Provider::default_model`] if unset.
    pub fn model(self, m: impl Into<String>) -> Self {
        Self {
            model: Some(m.into()),
            ..self
        }
    }

    /// Sets the system instruction (system prompt). Defaults to empty string.
    pub fn instruction(self, s: impl Into<String>) -> Self {
        Self {
            instruction: Some(s.into()),
            ..self
        }
    }

    /// Sets the token budget for the response. Falls back to [`Provider::default_max_tokens`].
    pub fn max_tokens(self, n: u32) -> Self {
        Self {
            max_tokens: Some(n),
            ..self
        }
    }

    /// Enables or disables extended thinking. Only applied to providers that
    /// support it; ignored otherwise.
    pub fn thinking(self, t: bool) -> Self {
        Self {
            thinking: Some(t),
            ..self
        }
    }

    /// Sets the expected output format. Defaults to [`ResponseShape::Text`].
    pub fn response_shape(self, r: ResponseShape) -> Self {
        Self {
            response_shape: Some(r),
            ..self
        }
    }

    /// Validates configuration and constructs the provider client.
    ///
    /// Returns [`LlmError::Config`] when required fields are missing.
    pub fn build(self) -> Result<Box<dyn LlmClient>, LlmError> {
        use crate::providers::{
            anthropic::AnthropicClient, gemini::GeminiClient, ollama::OllamaClient,
            openai::OpenAiClient,
        };

        let provider = self
            .provider
            .ok_or_else(|| LlmError::Config("provider is required".into()))?;

        if provider.needs_key() && self.api_key.as_deref().unwrap_or("").is_empty() {
            return Err(LlmError::Config(format!(
                "{} requires an api_key",
                provider.label()
            )));
        }

        if matches!(provider, Provider::Ollama) && self.base_url.is_none() {
            return Err(LlmError::Config("Ollama requires a base_url".into()));
        }

        let model = self
            .model
            .unwrap_or_else(|| provider.default_model().into());

        let max_tokens = self
            .max_tokens
            .unwrap_or_else(|| provider.default_max_tokens());

        let thinking = if provider.supports_thinking() {
            self.thinking.unwrap_or(false)
        } else {
            false
        };

        let config = ClientConfig {
            api_key: self.api_key.unwrap_or_default(),
            base_url: self.base_url,
            model,
            instruction: self.instruction.unwrap_or_default(),
            max_tokens,
            thinking,
            response_shape: self.response_shape.unwrap_or(ResponseShape::Text),
        };

        match provider {
            Provider::Anthropic => Ok(Box::new(AnthropicClient::new(config)?)),
            Provider::Gemini => Ok(Box::new(GeminiClient::new(config)?)),
            Provider::OpenAI => Ok(Box::new(OpenAiClient::new(config)?)),
            Provider::Ollama => Ok(Box::new(OllamaClient::new(config)?)),
        }
    }
}
