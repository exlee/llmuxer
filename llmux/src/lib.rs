pub mod builder;
pub mod config;
pub mod error;
pub mod providers;
pub mod traits;

pub use builder::{LlmClientBuilder, ResponseShape};
pub use config::{LlmConfig, Provider};
pub use error::LlmError;
pub use traits::{CacheResult, LlmClient, LlmClientExt};

#[cfg(test)]
mod tests {
    mod builder;
    mod config;
    mod provider;
    mod trait_defaults;
}
