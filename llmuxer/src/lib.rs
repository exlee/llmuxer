pub mod attachment;
pub mod builder;
pub mod config;
pub mod error;
pub mod shared;
pub mod token_extraction;
pub mod token_usage;

#[cfg(feature = "async")]
mod r#async;
#[cfg(not(feature = "async"))]
mod sync;

pub mod providers;

// Re-export the active traits module.
#[cfg(feature = "async")]
pub use r#async::traits;
#[cfg(not(feature = "async"))]
pub use sync::traits;

pub use attachment::{Attachment, AttachmentData};
pub use builder::{LlmClientBuilder, ResponseShape};
pub use config::{LlmConfig, Provider};
pub use error::LlmError;
pub use shared::CacheResult;
pub use token_usage::{TokenUsage, WithTokenUsage};
pub use traits::{CacheBuilder, LlmClient, QueryBuilder, QueryBuilderWithTokens};

#[cfg(test)]
mod tests {
    mod builder;
    mod config;
    mod provider;
    mod trait_defaults;
}
