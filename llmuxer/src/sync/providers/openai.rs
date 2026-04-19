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
        let base_url = match config.base_url {
            Some(url) => url,
            None => "https://api.openai.com".to_string(),
        };
        Ok(Self {
            api_key: config.api_key,
            base_url: base_url,
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

    fn build_user_content(prompt: &str, attachments: &[Attachment]) -> Result<Value, LlmError> {
        if attachments.is_empty() {
            return Ok(json!(prompt));
        }

        let mut parts: Vec<Value> = Vec::new();

        for att in attachments {
            let (bytes, mime) = att.resolve()?;
            if mime.starts_with("image/") {
                let data_url = format!("data:{mime};base64,{}", B64.encode(&bytes));
                parts.push(json!({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }));
            } else {
                return Err(LlmError::Config(format!(
                    "OpenAI does not support document attachments (mime: {mime})"
                )));
            }
        }

        parts.push(json!({"type": "text", "text": prompt}));
        Ok(Value::Array(parts))
    }

    fn build_body(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        system_prefix: Option<&str>,
    ) -> Result<Value, LlmError> {
        let system_content = match system_prefix {
            Some(prefix) => format!("{}\n\n{}", self.instruction, prefix),
            None => self.instruction.clone(),
        };

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": Self::build_user_content(prompt, attachments)?}
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

        Ok(body)
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

        let parsed: Value = serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
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
        let prefix = match cache {
            Some(CacheResult::Key(id)) => Some(id.as_str()),
            _ => None,
        };
        self.send_and_extract(self.build_body(prompt, attachments, prefix)?)
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache: Option<&CacheResult>,
    ) -> Result<WithTokenUsage<String>, LlmError> {
        let prefix = match cache {
            Some(CacheResult::Key(id)) => Some(id.as_str()),
            _ => None,
        };
        let body = self.build_body(prompt, attachments, prefix)?;
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

        let parsed: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
                reason: e.to_string(),
                raw: raw.clone(),
            })?;

        let token_usage = token_extraction::extract_openai(&parsed);
        let result = parsed["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::Deserialise {
                reason: "could not find text in choices[0].message.content".into(),
                raw,
            })?;

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
        // OpenAI prompt caching is automatic — no explicit cache creation endpoint.
        Ok(CacheResult::Key(content.to_string()))
    }
}
