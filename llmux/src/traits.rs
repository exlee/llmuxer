use serde::de::DeserializeOwned;

use crate::{attachment::Attachment, error::LlmError};

/// The outcome of a [`CacheBuilder::build`] call.
#[derive(Debug, Clone)]
pub enum CacheResult {
    /// A cache key to pass via [`QueryBuilder::cache`].
    Key(String),
    /// This provider does not support caching.
    Unsupported,
}

/// Fluent builder for executing a single query.
///
/// Constructed by [`LlmClient::query`]; not constructed directly.
///
/// The lifetime `'c` is tied to the `&'c dyn LlmClient` and the optional
/// `&'c CacheResult`, keeping the builder object-safe with no heap allocation.
pub struct QueryBuilder<'c> {
    client: &'c dyn LlmClient,
    prompt: String,
    attachments: Vec<Attachment>,
    cache: Option<&'c CacheResult>,
    require_cache: bool,
}

impl<'c> QueryBuilder<'c> {
    pub(crate) fn new(client: &'c dyn LlmClient, prompt: impl Into<String>) -> Self {
        Self {
            client,
            prompt: prompt.into(),
            attachments: Vec::new(),
            cache: None,
            require_cache: false,
        }
    }

    /// Append a single attachment.
    pub fn attachment(mut self, a: Attachment) -> Self {
        self.attachments.push(a);
        self
    }

    /// Append multiple attachments.
    pub fn attachments(mut self, a: impl IntoIterator<Item = Attachment>) -> Self {
        self.attachments.extend(a);
        self
    }

    /// Attach an existing cache result. Ignored if `Unsupported` unless
    /// [`require_cache`](Self::require_cache) is set.
    pub fn cache(self, c: &'c CacheResult) -> Self {
        Self {
            cache: Some(c),
            ..self
        }
    }

    /// If set, [`run`](Self::run) returns
    /// [`Err(LlmError::CacheRequired)`](LlmError::CacheRequired) when no
    /// cache is provided or the cache is [`CacheResult::Unsupported`].
    pub fn require_cache(self) -> Self {
        Self {
            require_cache: true,
            ..self
        }
    }

    /// Execute the query, returning raw response text.
    pub fn run(self) -> Result<String, LlmError> {
        if self.require_cache {
            match self.cache {
                None | Some(CacheResult::Unsupported) => return Err(LlmError::CacheRequired),
                Some(CacheResult::Key(_)) => {}
            }
        }
        self.client
            .execute_query(&self.prompt, &self.attachments, self.cache)
    }

    /// Execute the query and deserialize the response as `T`.
    pub fn json<T: DeserializeOwned>(self) -> Result<T, LlmError> {
        let raw = self.run()?;
        serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
            reason: e.to_string(),
            raw,
        })
    }
}

/// Fluent builder for pre-warming a cache entry.
///
/// Constructed by [`LlmClient::build_cache`]; not constructed directly.
pub struct CacheBuilder<'c> {
    client: &'c dyn LlmClient,
    content: String,
    attachments: Vec<Attachment>,
}

impl<'c> CacheBuilder<'c> {
    pub(crate) fn new(client: &'c dyn LlmClient, content: impl Into<String>) -> Self {
        Self {
            client,
            content: content.into(),
            attachments: Vec::new(),
        }
    }

    /// Append a single attachment to the cached content.
    pub fn attachment(mut self, a: Attachment) -> Self {
        self.attachments.push(a);
        self
    }

    /// Append multiple attachments.
    pub fn attachments(mut self, a: impl IntoIterator<Item = Attachment>) -> Self {
        self.attachments.extend(a);
        self
    }

    /// Submit the cache entry. Returns `Ok(CacheResult::Unsupported)` if the
    /// provider does not support caching, `Err` on failure.
    pub fn build(self) -> Result<CacheResult, LlmError> {
        self.client.execute_cache(&self.content, &self.attachments)
    }
}

/// Core synchronous client trait. Implement this to add a new provider.
///
/// Use [`LlmClientBuilder`](crate::LlmClientBuilder) to construct provider
/// clients. Call [`query`](LlmClient::query) or
/// [`build_cache`](LlmClient::build_cache) to begin a fluent interaction.
pub trait LlmClient: Send + Sync {
    /// Begin building a query against this client.
    fn query(&self, prompt: &str) -> QueryBuilder<'_>;

    /// Begin building a cache entry.
    fn build_cache(&self, content: &str) -> CacheBuilder<'_>;

    /// Execute a query. Called by [`QueryBuilder::run`]; not intended to be
    /// called directly.
    fn execute_query(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache: Option<&CacheResult>,
    ) -> Result<String, LlmError>;

    /// Execute a cache build. Called by [`CacheBuilder::build`]; not intended
    /// to be called directly.
    fn execute_cache(
        &self,
        content: &str,
        attachments: &[Attachment],
    ) -> Result<CacheResult, LlmError>;
}
