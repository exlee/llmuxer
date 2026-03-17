use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Provider {
    #[default]
    Anthropic,
    Gemini,
    OpenAI,
    Ollama,
}

impl Provider {
    /// Human-readable label for UI display.
    pub fn label(&self) -> &str {
        match self {
            Provider::Anthropic => "Anthropic",
            Provider::Gemini => "Gemini",
            Provider::OpenAI => "OpenAI",
            Provider::Ollama => "Ollama",
        }
    }

    /// Default model identifier.
    pub fn default_model(&self) -> &str {
        match self {
            Provider::Anthropic => "claude-sonnet-4-20250514",
            Provider::Gemini => "gemini-2.5-flash",
            Provider::OpenAI => "gpt-4o-mini",
            Provider::Ollama => "llama3",
        }
    }

    /// Default max_tokens.
    pub fn default_max_tokens(&self) -> u32 {
        match self {
            Provider::Anthropic => 8192,
            Provider::Gemini => 8192,
            Provider::OpenAI => 4096,
            Provider::Ollama => 4096,
        }
    }

    /// Whether an API key is required (false only for Ollama).
    pub fn needs_key(&self) -> bool {
        !matches!(self, Provider::Ollama)
    }

    /// Whether extended thinking is supported.
    pub fn supports_thinking(&self) -> bool {
        matches!(
            self,
            Provider::Anthropic | Provider::Gemini | Provider::OpenAI
        )
    }

    /// Whether context caching is supported.
    pub fn supports_caching(&self) -> bool {
        matches!(
            self,
            Provider::Anthropic | Provider::Gemini | Provider::OpenAI
        )
    }
}

/// Serialisable connectivity config — contains only what is needed to reach a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: Provider,
    pub api_key: String,
    /// Some for Ollama, None for all others.
    pub base_url: Option<String>,
    pub model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        let provider = Provider::default();
        Self {
            model: provider.default_model().into(),
            api_key: String::new(),
            base_url: None,
            provider,
        }
    }
}

impl LlmConfig {
    /// True if the config has enough information to build a client.
    /// Ollama: base_url must be Some. Others: api_key must be non-empty.
    pub fn is_ready(&self) -> bool {
        match self.provider {
            Provider::Ollama => self.base_url.is_some(),
            _ => !self.api_key.is_empty(),
        }
    }
}
