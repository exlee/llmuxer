use serde_json::{json, Value};

use crate::{
    builder::{ClientConfig, ResponseShape},
    error::LlmError,
    traits::{CacheResult, LlmClient},
};

pub struct OpenAiClient {
    api_key: String,
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    thinking: bool,
    response_shape: ResponseShape,
    client: reqwest::blocking::Client,
}

impl OpenAiClient {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        Ok(Self {
            api_key: config.api_key,
            base_url: "https://api.openai.com".into(),
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            thinking: config.thinking,
            response_shape: config.response_shape,
            client: reqwest::blocking::Client::new(),
        })
    }

    fn build_body(&self, query: &str, system_prefix: Option<&str>) -> Value {
        let system_content = match system_prefix {
            Some(prefix) => format!("{}\n\n{}", self.instruction, prefix),
            None => self.instruction.clone(),
        };

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        });

        if self.thinking {
            body["reasoning_effort"] = json!("high");
        }

        match &self.response_shape {
            ResponseShape::Text => {}
            ResponseShape::Json(schema) => {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": schema,
                        "strict": true
                    }
                });
            }
        }

        body
    }

    fn send_and_extract(&self, body: Value) -> Result<String, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let auth = format!("Bearer {}", self.api_key);

        let resp = self
            .client
            .post(&url)
            .header("Authorization", auth)
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

        parsed["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::Deserialise {
                reason: "could not find text in choices[0].message.content".into(),
                raw,
            })
    }
}

impl LlmClient for OpenAiClient {
    fn query(&self, query: &str) -> Result<String, LlmError> {
        self.send_and_extract(self.build_body(query, None))
    }

    fn query_with_cache(&self, query: &str, cache_id: &str) -> Result<String, LlmError> {
        self.send_and_extract(self.build_body(query, Some(cache_id)))
    }

    fn build_cache(&self, content: &str) -> CacheResult {
        // OpenAI prompt caching is automatic — no explicit cache creation endpoint.
        CacheResult::Key(content.to_string())
    }
}
