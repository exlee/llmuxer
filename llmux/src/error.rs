#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Network or transport failure from reqwest.
    #[error("provider request failed: {0}")]
    Request(#[from] reqwest::Error),

    /// The provider returned an HTTP error status.
    #[error("provider returned an error: status={status}, body={body}")]
    ProviderError { status: u16, body: String },

    /// The response could not be parsed into the expected shape.
    #[error("failed to deserialise response: {reason}\nraw: {raw}")]
    Deserialise { reason: String, raw: String },

    /// A cache operation failed.
    #[error("cache error: {0}")]
    Cache(String),

    /// A required configuration field is missing or invalid.
    #[error("configuration error: {0}")]
    Config(String),
}
