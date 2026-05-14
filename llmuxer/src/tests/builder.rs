#[cfg(test)]
mod tests {
    use crate::{LlmClientBuilder, LlmError, Provider, ReasoningEffort};

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

    // ── llama.cpp builder ─────────────────────────────────────────────

    #[test]
    fn llamacpp_builds_without_api_key() {
        let client = LlmClientBuilder::new()
            .provider(Provider::LlamaCpp)
            .build();
        assert!(
            client.is_ok(),
            "llama.cpp should build without an API key"
        );
    }

    #[test]
    fn llamacpp_builds_with_explicit_base_url() {
        let client = LlmClientBuilder::new()
            .provider(Provider::LlamaCpp)
            .base_url("http://myserver:9090")
            .build();
        assert!(client.is_ok());
    }

    // ── thinking_budget ───────────────────────────────────────────────

    #[test]
    fn thinking_budget_is_ignored_when_thinking_off() {
        // Should build fine; thinking_budget is silently zeroed when thinking=false
        let client = LlmClientBuilder::new()
            .provider(Provider::Anthropic)
            .api_key("sk-test")
            .thinking(false)
            .thinking_budget(4096)
            .build();
        assert!(client.is_ok());
    }

    #[test]
    fn thinking_budget_accepted_with_thinking_on() {
        let client = LlmClientBuilder::new()
            .provider(Provider::Anthropic)
            .api_key("sk-test")
            .thinking(true)
            .thinking_budget(4096)
            .build();
        assert!(client.is_ok());
    }

    // ── reasoning_effort ──────────────────────────────────────────────

    #[test]
    fn reasoning_effort_defaults_to_medium() {
        assert_eq!(ReasoningEffort::default(), ReasoningEffort::Medium);
    }

    #[test]
    fn reasoning_effort_accepted_on_openai() {
        let client = LlmClientBuilder::new()
            .provider(Provider::OpenAI)
            .api_key("sk-test")
            .thinking(true)
            .reasoning_effort(ReasoningEffort::Low)
            .build();
        assert!(client.is_ok());
    }

    #[test]
    fn reasoning_effort_accepted_on_llamacpp() {
        let client = LlmClientBuilder::new()
            .provider(Provider::LlamaCpp)
            .thinking(true)
            .reasoning_effort(ReasoningEffort::High)
            .build();
        assert!(client.is_ok());
    }
}
