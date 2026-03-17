use serde::de::DeserializeOwned;

use crate::error::LlmError;

/// The outcome of a cache-build attempt.
#[derive(Debug)]
pub enum CacheResult {
    /// A cache key that can be passed to [`LlmClient::query_with_cache`].
    Key(String),
    /// This provider does not support caching.
    Unsupported,
    /// Cache creation was attempted but failed.
    Err(LlmError),
}

/// Core synchronous client trait. Implement this to add a new provider.
///
/// Only [`query`](LlmClient::query) is required. The caching methods have
/// default implementations that fall back to a plain query.
pub trait LlmClient: Send + Sync {
    /// Send a prompt and return the raw response text.
    fn query(&self, query: &str) -> Result<String, LlmError>;

    /// Send a prompt using a previously built cache handle when available.
    ///
    /// Falls back to [`query`](LlmClient::query) when `cache` is
    /// [`Unsupported`](CacheResult::Unsupported) or [`Err`](CacheResult::Err).
    fn query_cached(&self, query: &str, cache: &CacheResult) -> Result<String, LlmError> {
        match cache {
            CacheResult::Key(id) => self.query_with_cache(query, id),
            CacheResult::Unsupported | CacheResult::Err(_) => self.query(query),
        }
    }

    /// Send a prompt with an explicit cache identifier.
    ///
    /// The default implementation ignores `cache_id` and delegates to
    /// [`query`](LlmClient::query). Providers that support context caching
    /// override this to attach the identifier to the request.
    fn query_with_cache(&self, query: &str, cache_id: &str) -> Result<String, LlmError> {
        let _ = cache_id;
        self.query(query)
    }

    /// Pre-warm a cache entry with the given content.
    ///
    /// Returns [`CacheResult::Unsupported`] by default. Providers that support
    /// server-side context caching (Anthropic, Gemini) override this to
    /// register the content and return a [`CacheResult::Key`].
    fn build_cache(&self, content: &str) -> CacheResult {
        let _ = content;
        CacheResult::Unsupported
    }
}

/// Extension trait providing typed JSON helpers.
///
/// Blanket-implemented for all [`LlmClient`] values. Not object-safe due to
/// generic return types; use [`LlmClient`] for `dyn` contexts.
pub trait LlmClientExt: LlmClient {
    /// Like [`LlmClient::query`] but deserialises the response as `T`.
    fn query_json<T>(&self, query: &str) -> Result<T, LlmError>
    where
        T: DeserializeOwned,
    {
        let raw = self.query(query)?;
        serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
            reason: e.to_string(),
            raw,
        })
    }

    /// Like [`LlmClient::query_cached`] but deserialises the response as `T`.
    fn query_json_cached<T>(&self, query: &str, cache: &CacheResult) -> Result<T, LlmError>
    where
        T: DeserializeOwned,
    {
        let raw = self.query_cached(query, cache)?;
        serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
            reason: e.to_string(),
            raw,
        })
    }
}

impl<C: LlmClient + ?Sized> LlmClientExt for C {}
