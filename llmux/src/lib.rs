pub mod attachment;
pub mod builder;
pub mod config;
pub mod error;
pub mod providers;
pub mod traits;

pub use attachment::{Attachment, AttachmentData};
pub use builder::{LlmClientBuilder, ResponseShape};
pub use config::{LlmConfig, Provider};
pub use error::LlmError;
pub use traits::{CacheBuilder, CacheResult, LlmClient, QueryBuilder};

#[cfg(test)]
mod tests {
    mod builder;
    mod config;
    mod provider;
    mod trait_defaults;
}
