/// The outcome of a [`CacheBuilder::build`](crate::CacheBuilder::build) call.
#[derive(Debug, Clone)]
pub enum CacheResult {
    /// A cache key to pass via [`QueryBuilder::cache`](crate::QueryBuilder::cache).
    Key(String),
    /// This provider does not support caching.
    Unsupported,
}
