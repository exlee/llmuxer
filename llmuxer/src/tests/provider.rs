#[cfg(test)]
mod tests {
    use crate::Provider;

    #[test]
    fn ollama_does_not_need_key() {
        assert!(!Provider::Ollama.needs_key());
    }

    #[test]
    fn cloud_providers_need_key() {
        for p in [Provider::Anthropic, Provider::Gemini, Provider::OpenAI] {
            assert!(p.needs_key(), "{p:?} should need a key");
        }
    }

    #[test]
    fn ollama_does_not_support_caching_or_thinking() {
        assert!(!Provider::Ollama.supports_caching());
        assert!(!Provider::Ollama.supports_thinking());
    }
}
