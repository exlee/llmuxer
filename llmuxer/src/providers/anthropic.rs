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
pub type AnthropicProvider = AnthropicClient<reqwest::blocking::Client>;
#[cfg(feature = "async")]
pub type AnthropicProvider = AnthropicClient<reqwest::Client>;

// ---- generic struct + shared logic ----

pub struct AnthropicClient<C> {
    api_key: String,
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    thinking: bool,
    response_shape: ResponseShape,
    http: C,
}

impl<C> AnthropicClient<C> {
    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-api-key", self.api_key.parse().unwrap());
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());
        if self.thinking {
            headers.insert(
                "anthropic-beta",
                "interleaved-thinking-2025-05-14".parse().unwrap(),
            );
        }
        headers
    }

    fn build_user_content(prompt: &str, attachments: &[Attachment]) -> Result<Value, LlmError> {
        if attachments.is_empty() {
            return Ok(json!(prompt));
        }

        let mut parts: Vec<Value> = Vec::new();

        for att in attachments {
            let (bytes, mime) = att.resolve()?;
            let block = if mime.starts_with("image/") {
                json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": B64.encode(&bytes)
                    }
                })
            } else if mime == "text/plain" {
                json!({
                    "type": "document",
                    "source": {
                        "type": "text",
                        "data": String::from_utf8_lossy(&bytes).into_owned()
                    }
                })
            } else {
                json!({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": B64.encode(&bytes)
                    }
                })
            };
            parts.push(block);
        }

        parts.push(json!({"type": "text", "text": prompt}));
        Ok(Value::Array(parts))
    }

    fn build_body(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache_id: Option<&str>,
    ) -> Result<Value, LlmError> {
        let system: Value = match cache_id {
            None => json!(self.instruction),
            Some(id) => json!([
                {"type": "text", "text": self.instruction},
                {"type": "text", "text": id, "cache_control": {"type": "ephemeral"}}
            ]),
        };

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{
                "role": "user",
                "content": Self::build_user_content(prompt, attachments)?
            }]
        });

        if self.thinking {
            let budget = (self.max_tokens / 2).max(1024);
            body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
        }

        self.apply_response_shape(&mut body);
        Ok(body)
    }

    fn apply_response_shape(&self, body: &mut Value) {
        if let ResponseShape::Json(schema) = &self.response_shape {
            body["tools"] = json!([{
                "name": "respond",
                "description": "Provide your structured response",
                "input_schema": schema
            }]);
            body["tool_choice"] = json!({"type": "tool", "name": "respond"});
        }
    }

    fn extract_text(&self, parsed: &Value, raw: &str) -> Result<String, LlmError> {
        let content = parsed["content"]
            .as_array()
            .ok_or_else(|| LlmError::Deserialise {
                reason: "missing content array".into(),
                raw: raw.to_string(),
            })?;

        match &self.response_shape {
            ResponseShape::Json(_) => {
                for block in content {
                    if block["type"] == "tool_use" {
                        return Ok(block["input"].to_string());
                    }
                }
                Err(LlmError::Deserialise {
                    reason: "no tool_use block in response".into(),
                    raw: raw.to_string(),
                })
            }
            ResponseShape::Text => {
                let text: String = content
                    .iter()
                    .filter(|b| b["type"] == "text")
                    .filter_map(|b| b["text"].as_str())
                    .collect();

                if text.is_empty() {
                    Err(LlmError::Deserialise {
                        reason: "no text block in response".into(),
                        raw: raw.to_string(),
                    })
                } else {
                    Ok(text)
                }
            }
        }
    }
}

// ---- sync implementation ----

#[cfg(not(feature = "async"))]
use crate::traits::LlmClient;

#[cfg(not(feature = "async"))]
impl AnthropicClient<reqwest::blocking::Client> {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        Ok(Self {
            api_key: config.api_key,
            base_url: "https://api.anthropic.com".into(),
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            thinking: config.thinking,
            response_shape: config.response_shape,
            http: reqwest::blocking::Client::builder()
                .timeout(config.timeout)
                .build()?,
        })
    }

    fn send_request(&self, body: Value) -> Result<(Value, String), LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .http
            .post(&url)
            .headers(self.headers())
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
impl LlmClient for AnthropicClient<reqwest::blocking::Client> {
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
        cache: Option<&CacheResult>,
    ) -> Result<String, LlmError> {
        let cache_id = match cache {
            Some(CacheResult::Key(id)) => Some(id.as_str()),
            _ => None,
        };
        self.send_and_extract(self.build_body(prompt, attachments, cache_id)?)
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache: Option<&CacheResult>,
    ) -> Result<WithTokenUsage<String>, LlmError> {
        let cache_id = match cache {
            Some(CacheResult::Key(id)) => Some(id.as_str()),
            _ => None,
        };
        let body = self.build_body(prompt, attachments, cache_id)?;
        let (parsed, raw) = self.send_request(body)?;
        let token_usage = token_extraction::extract_anthropic(&parsed);
        let result = self.extract_text(&parsed, &raw)?;
        Ok(WithTokenUsage {
            token_usage,
            result,
        })
    }

    fn execute_cache(
        &self,
        content: &str,
        _attachments: &[Attachment],
    ) -> Result<CacheResult, LlmError> {
        Ok(CacheResult::Key(content.to_string()))
    }
}

// ---- async implementation ----

#[cfg(feature = "async")]
use crate::traits::LlmClient;
#[cfg(feature = "async")]
use futures::future::BoxFuture;

#[cfg(feature = "async")]
impl AnthropicClient<reqwest::Client> {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        Ok(Self {
            api_key: config.api_key,
            base_url: "https://api.anthropic.com".into(),
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            thinking: config.thinking,
            response_shape: config.response_shape,
            http: reqwest::Client::builder().timeout(config.timeout).build()?,
        })
    }

    async fn send_request(&self, body: Value) -> Result<(Value, String), LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .http
            .post(&url)
            .headers(self.headers())
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
impl LlmClient for AnthropicClient<reqwest::Client> {
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
        cache: Option<&CacheResult>,
    ) -> BoxFuture<'_, Result<String, LlmError>> {
        let prompt = prompt.to_owned();
        let attachments = attachments.to_vec();
        let cache_key = cache.and_then(|c| match c {
            CacheResult::Key(id) => Some(id.clone()),
            _ => None,
        });

        Box::pin(async move {
            let cache_id = cache_key.as_deref();
            let body = self.build_body(&prompt, &attachments, cache_id)?;
            self.send_and_extract(body).await
        })
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache: Option<&CacheResult>,
    ) -> BoxFuture<'_, Result<WithTokenUsage<String>, LlmError>> {
        let prompt = prompt.to_owned();
        let attachments = attachments.to_vec();
        let cache_key = cache.and_then(|c| match c {
            CacheResult::Key(id) => Some(id.clone()),
            _ => None,
        });

        Box::pin(async move {
            let cache_id = cache_key.as_deref();
            let body = self.build_body(&prompt, &attachments, cache_id)?;
            let (parsed, raw) = self.send_request(body).await?;
            let token_usage = token_extraction::extract_anthropic(&parsed);
            let result = self.extract_text(&parsed, &raw)?;
            Ok(WithTokenUsage {
                token_usage,
                result,
            })
        })
    }

    fn execute_cache(
        &self,
        content: &str,
        _attachments: &[Attachment],
    ) -> BoxFuture<'_, Result<CacheResult, LlmError>> {
        let content = content.to_owned();
        Box::pin(async move { Ok(CacheResult::Key(content)) })
    }
}
