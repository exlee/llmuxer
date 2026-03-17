// llmuxer/src/tests/config.rs
#[cfg(test)]
mod tests {
    use crate::{LlmConfig, Provider};

    #[test]
    fn is_ready_false_when_api_key_empty() {
        let config = LlmConfig::default(); // Anthropic, empty key
        assert!(!config.is_ready());
    }

    #[test]
    fn is_ready_true_when_api_key_set() {
        let config = LlmConfig {
            provider: Provider::Anthropic,
            api_key: "sk-test".into(),
            base_url: None,
            model: "claude-sonnet-4-20250514".into(),
        };
        assert!(config.is_ready());
    }

    #[test]
    fn is_ready_for_ollama_requires_base_url() {
        let mut config = LlmConfig {
            provider: Provider::Ollama,
            api_key: String::new(),
            base_url: None,
            model: "llama3".into(),
        };
        assert!(!config.is_ready());
        config.base_url = Some("http://localhost:11434".into());
        assert!(config.is_ready());
    }
}
