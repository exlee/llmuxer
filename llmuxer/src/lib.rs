pub mod attachment;
pub mod builder;
pub mod config;
pub mod error;
#[cfg(not(feature = "async"))]
include!("lib_sync.rs");
#[cfg(feature = "async")]
include!("lib_async.rs");

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
