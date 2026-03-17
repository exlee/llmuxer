# llmuxer-keystore

Credential persistence for llmuxer. Stores one `LlmConfig` per provider in `~/.config/llmuxer/config.json` with file permissions set to 0600.

## Usage

```rust
use llmuxer::{LlmConfig, Provider};
use llmuxer_keystore::ProviderStore;

// Save credentials
let mut store = ProviderStore::load()?;
store.set(Provider::Anthropic, LlmConfig {
    provider: Provider::Anthropic,
    api_key: "sk-ant-...".into(),
    model: "claude-sonnet-4-20250514".into(),
    base_url: None,
});
store.save()?;

// Load and use
let store = ProviderStore::load()?;
if let Some(config) = store.get(&Provider::Anthropic) {
    let client = llmuxer::LlmClientBuilder::new()
        .config(config.clone())
        .build()?;
}
```

The config file is created automatically. Parent directories are created as needed.
