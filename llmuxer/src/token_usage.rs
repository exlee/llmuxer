/// Input/output token counters returned by a provider when the
/// [`QueryBuilder::with_tokens`](crate::traits::QueryBuilder::with_tokens)
/// path is used.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenUsage {
    /// Complete input token count (prompt + system + attachments).
    pub prompt_token_count: Option<usize>,
    /// Cached input token count — tokens served from a provider-side cache.
    pub cached_content_token_count: Option<usize>,
    /// Token count consumed by model "thinking" / reasoning.
    pub thoughts_token_count: Option<usize>,
    /// Output token count (completion).
    pub output_token_count: Option<usize>,
    /// Total token count — typically input + output.
    pub total_token_count: Option<usize>,
}

impl TokenUsage {
    /// Convenience constructor that returns a `TokenUsage` with all fields
    /// set to `None`.
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Wrapper that pairs a result value with the token usage reported by the
/// provider for the request that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WithTokenUsage<T> {
    pub token_usage: TokenUsage,
    pub result: T,
}
