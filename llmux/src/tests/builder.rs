#[cfg(test)]
mod tests {
    use crate::{LlmClientBuilder, LlmError, Provider};

    #[test]
    fn build_fails_without_provider() {
        let err = LlmClientBuilder::new()
            .api_key("key")
            .instruction("hi")
            .build()
            .err()
            .unwrap();

        assert!(matches!(err, LlmError::Config(_)));
    }

    #[test]
    fn build_fails_without_api_key_for_anthropic() {
        let err = LlmClientBuilder::new()
            .provider(Provider::Anthropic)
            .instruction("hi")
            .build()
            .err()
            .unwrap();
        assert!(matches!(err, LlmError::Config(_)));
    }

    #[test]
    fn build_fails_without_base_url_for_ollama() {
        let err = LlmClientBuilder::new()
            .provider(Provider::Ollama)
            .instruction("hi")
            .build()
            .err()
            .unwrap();
        assert!(matches!(err, LlmError::Config(_)));
    }

    #[test]
    fn build_succeeds_for_ollama_with_base_url() {
        let client = LlmClientBuilder::new()
            .provider(Provider::Ollama)
            .base_url("http://localhost:11434")
            .instruction("hi")
            .build();
        assert!(client.is_ok());
    }

    #[test]
    fn model_defaults_to_provider_default() {
        // Build an Ollama client (no HTTP), inspect what model was baked in.
        // Requires either making OllamaClient fields accessible via #[cfg(test)]
        // or adding a model() accessor to LlmClient.
    }
}
