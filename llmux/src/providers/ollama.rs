use serde_json::{json, Value};

use crate::{
    builder::{ClientConfig, ResponseShape},
    error::LlmError,
    traits::LlmClient,
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
            client: reqwest::blocking::Client::new(),
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
}

impl LlmClient for OllamaClient {
    fn query(&self, query: &str) -> Result<String, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let body = json!({
            "model": self.model,
            "stream": false,
            "options": {"num_predict": self.max_tokens},
            "messages": [
                {"role": "system", "content": self.build_system()},
                {"role": "user", "content": query}
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

        let parsed: Value =
            serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
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
    // build_cache and query_with_cache inherit default Unsupported implementations.
}
