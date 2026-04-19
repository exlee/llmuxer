use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

use crate::{
    attachment::Attachment,
    builder::{ClientConfig, ResponseShape},
    error::LlmError,
    token_extraction,
    token_usage::WithTokenUsage,
    traits::{CacheBuilder, CacheResult, LlmClient, QueryBuilder},
};

pub struct AnthropicClient {
    api_key: String,
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    thinking: bool,
    response_shape: ResponseShape,
    client: reqwest::blocking::Client,
}

impl AnthropicClient {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        Ok(Self {
            api_key: config.api_key,
            base_url: "https://api.anthropic.com".into(),
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            thinking: config.thinking,
            response_shape: config.response_shape,
            client: reqwest::blocking::Client::builder()
                .timeout(config.timeout)
                .build()?,
        })
    }

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

    /// Build the content array for a user message, including any attachments.
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
                // PDF and other document types
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

    fn send_and_extract(&self, body: Value) -> Result<String, LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .client
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

        self.extract_text(&parsed, &raw)
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

impl LlmClient for AnthropicClient {
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
        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()?;

        let status = resp.status().as_u16();
        let raw = resp.text()?;

        if status >= 400 {
            return Err(LlmError::ProviderError { status, body: raw });
        }

        let parsed: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
                reason: e.to_string(),
                raw: raw.clone(),
            })?;

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
        // Anthropic's ephemeral caching is marker-based: the content string
        // is stored as the key and replayed in the system message with
        // cache_control on each request.
        Ok(CacheResult::Key(content.to_string()))
    }
}
