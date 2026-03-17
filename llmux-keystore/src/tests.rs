#[cfg(test)]
mod tests {
    use crate::{KeystoreError, ProviderStore};
    use llmux::{LlmConfig, Provider};
    use std::path::{Path, PathBuf};

    /// Minimal RAII temp directory — no `tempfile` crate required.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("llmux_ks_test_{}_{}", std::process::id(), n));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn store_at(dir: &TempDir) -> ProviderStore {
        ProviderStore::load_from(dir.path().join("config.json")).unwrap()
    }

    #[test]
    fn load_returns_empty_store_when_file_absent() {
        let dir = TempDir::new();
        let store = store_at(&dir);
        assert!(store.configs.is_empty());
    }

    #[test]
    fn roundtrip_single_provider() {
        let dir = TempDir::new();
        let path = dir.path().join("config.json");

        let mut store = ProviderStore::load_from(&path).unwrap();
        store.set(
            Provider::Anthropic,
            LlmConfig {
                provider: Provider::Anthropic,
                api_key: "sk-test".into(),
                base_url: None,
                model: "claude-sonnet-4-20250514".into(),
            },
        );
        store.save_to(&path).unwrap();

        let loaded = ProviderStore::load_from(&path).unwrap();
        let config = loaded.get(&Provider::Anthropic).unwrap();
        assert_eq!(config.api_key, "sk-test");
    }

    #[test]
    fn roundtrip_all_providers() {
        let dir = TempDir::new();
        let path = dir.path().join("config.json");

        let mut store = ProviderStore::load_from(&path).unwrap();
        for provider in [Provider::Anthropic, Provider::Gemini, Provider::OpenAI] {
            store.set(
                provider.clone(),
                LlmConfig {
                    api_key: format!("key-for-{provider:?}"),
                    model: provider.default_model().into(),
                    base_url: None,
                    provider,
                },
            );
        }
        store.set(
            Provider::Ollama,
            LlmConfig {
                provider: Provider::Ollama,
                api_key: String::new(),
                base_url: Some("http://localhost:11434".into()),
                model: "llama3".into(),
            },
        );
        store.save_to(&path).unwrap();

        let loaded = ProviderStore::load_from(&path).unwrap();
        assert_eq!(loaded.configs.len(), 4);
        assert_eq!(
            loaded.get(&Provider::Gemini).unwrap().api_key,
            "key-for-Gemini"
        );
    }

    #[test]
    fn save_sets_file_permissions_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new();
        let path = dir.path().join("config.json");

        let store = ProviderStore::load_from(&path).unwrap();
        store.save_to(&path).unwrap();

        let perms = std::fs::metadata(&path).unwrap().permissions();
        assert_eq!(perms.mode() & 0o777, 0o600);
    }

    #[test]
    fn load_returns_error_on_corrupt_json() {
        let dir = TempDir::new();
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"not json at all").unwrap();

        let err = ProviderStore::load_from(&path).err().unwrap();
        assert!(matches!(err, KeystoreError::Deserialise(_)));
    }

    #[test]
    fn set_overwrites_existing_provider() {
        let dir = TempDir::new();
        let path = dir.path().join("config.json");

        let mut store = ProviderStore::load_from(&path).unwrap();
        store.set(
            Provider::Anthropic,
            LlmConfig {
                api_key: "old".into(),
                ..Default::default()
            },
        );
        store.save_to(&path).unwrap();

        let mut store = ProviderStore::load_from(&path).unwrap();
        store.set(
            Provider::Anthropic,
            LlmConfig {
                api_key: "new".into(),
                ..Default::default()
            },
        );
        store.save_to(&path).unwrap();

        let loaded = ProviderStore::load_from(&path).unwrap();
        assert_eq!(loaded.get(&Provider::Anthropic).unwrap().api_key, "new");
    }
}
