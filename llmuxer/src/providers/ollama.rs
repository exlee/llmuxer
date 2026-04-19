use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

use crate::{
    attachment::Attachment,
    builder::{ClientConfig, ResponseShape},
    error::LlmError,
    shared::CacheResult,
    token_extraction,
    token_usage::WithTokenUsage,
};

// ---- public type alias for the active mode ----

#[cfg(not(feature = "async"))]
pub type OllamaProvider = OllamaClient<reqwest::blocking::Client>;
#[cfg(feature = "async")]
pub type OllamaProvider = OllamaClient<reqwest::Client>;

// ---- generic struct + shared logic ----

pub struct OllamaClient<C> {
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    response_shape: ResponseShape,
    http: C,
}

impl<C> OllamaClient<C> {
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

    fn build_body(&self, prompt: &str, attachments: &[Attachment]) -> Result<Value, LlmError> {
        let images = Self::collect_images(attachments)?;

        let mut user_msg = json!({
            "role": "user",
            "content": prompt
        });
        if !images.is_empty() {
            user_msg["images"] = json!(images);
        }

        Ok(json!({
            "model": self.model,
            "stream": false,
            "options": {"num_predict": self.max_tokens},
            "messages": [
                {"role": "system", "content": self.build_system()},
                user_msg
            ]
        }))
    }

    fn extract_text(&self, parsed: &Value, raw: &str) -> Result<String, LlmError> {
        parsed["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::Deserialise {
                reason: "could not find text in message.content".into(),
                raw: raw.to_string(),
            })
    }
}

// ---- sync implementation ----

#[cfg(not(feature = "async"))]
use crate::traits::LlmClient;

#[cfg(not(feature = "async"))]
impl OllamaClient<reqwest::blocking::Client> {
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
            http: reqwest::blocking::Client::builder()
                .timeout(config.timeout)
                .build()?,
        })
    }

    fn send_request(&self, body: Value) -> Result<(Value, String), LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let resp = self
            .http
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

        Ok((parsed, raw))
    }

    fn send_and_extract(&self, body: Value) -> Result<String, LlmError> {
        let (parsed, raw) = self.send_request(body)?;
        self.extract_text(&parsed, &raw)
    }
}

#[cfg(not(feature = "async"))]
impl LlmClient for OllamaClient<reqwest::blocking::Client> {
    fn query(&self, prompt: &str) -> crate::traits::QueryBuilder<'_> {
        crate::traits::QueryBuilder::new(self, prompt)
    }

    fn build_cache(&self, content: &str) -> crate::traits::CacheBuilder<'_> {
        crate::traits::CacheBuilder::new(self, content)
    }

    fn execute_query(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        _cache: Option<&CacheResult>,
    ) -> Result<String, LlmError> {
        self.send_and_extract(self.build_body(prompt, attachments)?)
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        _cache: Option<&CacheResult>,
    ) -> Result<WithTokenUsage<String>, LlmError> {
        let body = self.build_body(prompt, attachments)?;
        let (parsed, raw) = self.send_request(body)?;
        let token_usage = token_extraction::extract_ollama(&parsed);
        let result = self.extract_text(&parsed, &raw)?;
        Ok(WithTokenUsage {
            token_usage,
            result,
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

// ---- async implementation ----

#[cfg(feature = "async")]
use crate::traits::LlmClient;
#[cfg(feature = "async")]
use futures::future::BoxFuture;

#[cfg(feature = "async")]
impl OllamaClient<reqwest::Client> {
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
            http: reqwest::Client::builder().timeout(config.timeout).build()?,
        })
    }

    async fn send_request(&self, body: Value) -> Result<(Value, String), LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let resp = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status().as_u16();
        let raw = resp.text().await?;

        if status >= 400 {
            return Err(LlmError::ProviderError { status, body: raw });
        }

        let parsed: Value = serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
            reason: e.to_string(),
            raw: raw.clone(),
        })?;

        Ok((parsed, raw))
    }

    async fn send_and_extract(&self, body: Value) -> Result<String, LlmError> {
        let (parsed, raw) = self.send_request(body).await?;
        self.extract_text(&parsed, &raw)
    }
}

#[cfg(feature = "async")]
impl LlmClient for OllamaClient<reqwest::Client> {
    fn query(&self, prompt: &str) -> crate::traits::QueryBuilder<'_> {
        crate::traits::QueryBuilder::new(self, prompt)
    }

    fn build_cache(&self, content: &str) -> crate::traits::CacheBuilder<'_> {
        crate::traits::CacheBuilder::new(self, content)
    }

    fn execute_query(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        _cache: Option<&CacheResult>,
    ) -> BoxFuture<'_, Result<String, LlmError>> {
        let prompt = prompt.to_string();
        let attachments = attachments.to_vec();

        Box::pin(async move {
            self.send_and_extract(self.build_body(&prompt, &attachments)?)
                .await
        })
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        _cache: Option<&CacheResult>,
    ) -> BoxFuture<'_, Result<WithTokenUsage<String>, LlmError>> {
        let prompt = prompt.to_string();
        let attachments = attachments.to_vec();

        Box::pin(async move {
            let body = self.build_body(&prompt, &attachments)?;
            let (parsed, raw) = self.send_request(body).await?;
            let token_usage = token_extraction::extract_ollama(&parsed);
            let result = self.extract_text(&parsed, &raw)?;
            Ok(WithTokenUsage {
                token_usage,
                result,
            })
        })
    }

    fn execute_cache(
        &self,
        _content: &str,
        _attachments: &[Attachment],
    ) -> BoxFuture<'_, Result<CacheResult, LlmError>> {
        Box::pin(async move { Ok(CacheResult::Unsupported) })
    }
}
