use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::os::unix::fs::PermissionsExt;
#[cfg(test)]
mod tests;
use llmuxer::{LlmConfig, Provider};

/// Errors that can occur during keystore operations.
#[derive(Debug)]
pub enum KeystoreError {
    /// The platform config directory could not be determined.
    NoConfigDir,
    /// An I/O error occurred while reading or writing the config file.
    Io(std::io::Error),
    /// The config file could not be parsed as JSON.
    Deserialise(serde_json::Error),
}

impl fmt::Display for KeystoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeystoreError::NoConfigDir => write!(f, "could not determine config directory"),
            KeystoreError::Io(e) => write!(f, "io error: {e}"),
            KeystoreError::Deserialise(e) => write!(f, "deserialise error: {e}"),
        }
    }
}

impl std::error::Error for KeystoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KeystoreError::Io(e) => Some(e),
            KeystoreError::Deserialise(e) => Some(e),
            KeystoreError::NoConfigDir => None,
        }
    }
}

impl From<std::io::Error> for KeystoreError {
    fn from(e: std::io::Error) -> Self {
        KeystoreError::Io(e)
    }
}

impl From<serde_json::Error> for KeystoreError {
    fn from(e: serde_json::Error) -> Self {
        KeystoreError::Deserialise(e)
    }
}

/// Shared credential store. Persists one `LlmConfig` per provider at
/// `~/.config/llmuxer/config.json` with permissions 0600.
pub struct ProviderStore {
    pub configs: HashMap<Provider, LlmConfig>,
}

impl ProviderStore {
    fn config_path() -> Result<std::path::PathBuf, KeystoreError> {
        let dir = dirs::config_dir().ok_or(KeystoreError::NoConfigDir)?;
        Ok(dir.join("llmuxer").join("config.json"))
    }

    /// Load from `~/.config/llmuxer/config.json`.
    /// Returns an empty store if the file does not exist.
    pub fn load() -> Result<Self, KeystoreError> {
        Self::load_from(Self::config_path()?)
    }

    /// Load from an explicit path. Returns an empty store if the file does not exist.
    pub fn load_from(path: impl AsRef<std::path::Path>) -> Result<Self, KeystoreError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self {
                configs: HashMap::new(),
            });
        }
        let contents = fs::read_to_string(path)?;
        let configs: HashMap<Provider, LlmConfig> = serde_json::from_str(&contents)?;
        Ok(Self { configs })
    }

    /// Write to `~/.config/llmuxer/config.json` with permissions 0600.
    pub fn save(&self) -> Result<(), KeystoreError> {
        self.save_to(Self::config_path()?)
    }

    /// Write to an explicit path with permissions 0600.
    pub fn save_to(&self, path: impl AsRef<std::path::Path>) -> Result<(), KeystoreError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.configs)?;
        fs::write(path, &json)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(())
    }

    /// Returns the config for the given provider, if one has been stored.
    pub fn get(&self, provider: &Provider) -> Option<&LlmConfig> {
        self.configs.get(provider)
    }

    /// Inserts or replaces the config for the given provider.
    pub fn set(&mut self, provider: Provider, config: LlmConfig) {
        self.configs.insert(provider, config);
    }
}
