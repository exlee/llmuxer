use std::path::PathBuf;

use crate::error::LlmError;

/// The payload of an [`Attachment`].
#[derive(Debug, Clone)]
pub enum AttachmentData {
    /// Read from disk at execution time; MIME type inferred from extension.
    Path(PathBuf),
    /// Pre-loaded bytes with an explicit MIME type.
    Bytes { data: Vec<u8>, mime_type: String },
}

/// A file or binary blob to include in a query or cache entry.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub data: AttachmentData,
    pub label: Option<String>,
}

impl Attachment {
    /// Construct from a filesystem path. MIME type is inferred from the file
    /// extension at execution time.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            data: AttachmentData::Path(path.into()),
            label: None,
        }
    }

    /// Construct from raw bytes with an explicit MIME type.
    pub fn from_bytes(data: Vec<u8>, mime_type: impl Into<String>) -> Self {
        Self {
            data: AttachmentData::Bytes {
                data,
                mime_type: mime_type.into(),
            },
            label: None,
        }
    }

    /// Set an optional name hint. Used by providers that identify uploaded
    /// files (e.g. the Gemini Files API).
    pub fn label(self, label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            ..self
        }
    }

    /// Resolve attachment to `(bytes, mime_type)`.
    pub(crate) fn resolve(&self) -> Result<(Vec<u8>, String), LlmError> {
        match &self.data {
            AttachmentData::Path(path) => {
                let data = std::fs::read(path).map_err(|e| {
                    LlmError::Config(format!("failed to read attachment {}: {e}", path.display()))
                })?;
                let mime = mime_from_ext(path);
                Ok((data, mime))
            }
            AttachmentData::Bytes { data, mime_type } => Ok((data.clone(), mime_type.clone())),
        }
    }
}

pub(crate) fn mime_from_ext(path: &std::path::Path) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("pdf") => "application/pdf".into(),
        Some("png") => "image/png".into(),
        Some("jpg") | Some("jpeg") => "image/jpeg".into(),
        Some("gif") => "image/gif".into(),
        Some("webp") => "image/webp".into(),
        Some("txt") | Some("md") => "text/plain".into(),
        _ => "application/octet-stream".into(),
    }
}
