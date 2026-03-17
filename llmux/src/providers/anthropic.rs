use serde_json::{json, Value};

use crate::{
    builder::{ClientConfig, ResponseShape},
    error::LlmError,
    traits::{CacheResult, LlmClient},
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
            client: reqwest::blocking::Client::new(),
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

    fn build_messages_body(&self, query: &str) -> Value {
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.instruction,
            "messages": [{"role": "user", "content": query}]
        });

        if self.thinking {
            let budget = (self.max_tokens / 2).max(1024);
            body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
        }

        self.apply_response_shape(&mut body);
        body
    }

    fn build_cached_body(&self, query: &str, cache_id: &str) -> Value {
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {"type": "text", "text": self.instruction},
                {"type": "text", "text": cache_id, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": query}]
        });

        if self.thinking {
            let budget = (self.max_tokens / 2).max(1024);
            body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
        }

        self.apply_response_shape(&mut body);
        body
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

        let parsed: Value =
            serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
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
    fn query(&self, query: &str) -> Result<String, LlmError> {
        self.send_and_extract(self.build_messages_body(query))
    }

    fn query_with_cache(&self, query: &str, cache_id: &str) -> Result<String, LlmError> {
        self.send_and_extract(self.build_cached_body(query, cache_id))
    }

    fn build_cache(&self, content: &str) -> CacheResult {
        CacheResult::Key(content.to_string())
    }
}
