use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

use crate::{
    attachment::Attachment,
    builder::{ClientConfig, ResponseShape},
    error::LlmError,
    traits::{CacheBuilder, CacheResult, LlmClient, QueryBuilder},
};

pub struct OllamaClient {
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    response_shape: ResponseShape,
    client: reqwest::blocking::Client,
}

impl OllamaClient {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        let base_url = config
            .base_url
            .ok_or_else(|| LlmError::Config("Ollama requires a base_url".into()))?;
        Ok(Self {
            base_url,
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            response_shape: config.response_shape,
            client: reqwest::blocking::Client::builder()
                .timeout(config.timeout)
                .build()?,
        })
    }

    fn build_system(&self) -> String {
        match &self.response_shape {
            ResponseShape::Text => self.instruction.clone(),
            ResponseShape::Json(schema) => format!(
                "{}\n\nRespond with JSON matching this schema:\n{}",
                self.instruction,
                serde_json::to_string_pretty(schema).unwrap_or_default()
            ),
        }
    }

    /// Collect base64-encoded image bytes from attachments. Non-image
    /// attachments return an error.
    fn collect_images(attachments: &[Attachment]) -> Result<Vec<String>, LlmError> {
        let mut images = Vec::new();
        for att in attachments {
            let (bytes, mime) = att.resolve()?;
            if mime.starts_with("image/") {
                images.push(B64.encode(&bytes));
            } else {
                return Err(LlmError::Config(format!(
                    "Ollama does not support document attachments (mime: {mime})"
                )));
            }
        }
        Ok(images)
    }
}

impl LlmClient for OllamaClient {
    fn query(&self, prompt: &str) -> QueryBuilder<'_> {
        let client: &dyn LlmClient = self;
        QueryBuilder::new(client, prompt)
    }

    fn build_cache(&self, content: &str) -> CacheBuilder<'_> {
        let client: &dyn LlmClient = self;
        CacheBuilder::new(client, content)
    }

    fn execute_query(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        _cache: Option<&CacheResult>,
    ) -> Result<String, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let images = Self::collect_images(attachments)?;

        let mut user_msg = json!({
            "role": "user",
            "content": prompt
        });
        if !images.is_empty() {
            user_msg["images"] = json!(images);
        }

        let body = json!({
            "model": self.model,
            "stream": false,
            "options": {"num_predict": self.max_tokens},
            "messages": [
                {"role": "system", "content": self.build_system()},
                user_msg
            ]
        });

        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()?;

        let status = resp.status().as_u16();
        let raw = resp.text()?;

        if status >= 400 {
            return Err(LlmError::ProviderError { status, body: raw });
        }

        let parsed: Value = serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
            reason: e.to_string(),
            raw: raw.clone(),
        })?;

        parsed["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::Deserialise {
                reason: "could not find text in message.content".into(),
                raw,
            })
    }

    fn execute_cache(
        &self,
        _content: &str,
        _attachments: &[Attachment],
    ) -> Result<CacheResult, LlmError> {
        Ok(CacheResult::Unsupported)
    }
}
