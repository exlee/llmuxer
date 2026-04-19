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
pub type OpenAiProvider = OpenAiClient<reqwest::blocking::Client>;
#[cfg(feature = "async")]
pub type OpenAiProvider = OpenAiClient<reqwest::Client>;

// ---- generic struct + shared logic ----

pub struct OpenAiClient<C> {
    api_key: String,
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    thinking: bool,
    response_shape: ResponseShape,
    http: C,
}

impl<C> OpenAiClient<C> {
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

    fn extract_text(&self, parsed: &Value, raw: &str) -> Result<String, LlmError> {
        parsed["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::Deserialise {
                reason: "could not find text in choices[0].message.content".into(),
                raw: raw.to_string(),
            })
    }
}

// ---- sync implementation ----

#[cfg(not(feature = "async"))]
use crate::traits::LlmClient;

#[cfg(not(feature = "async"))]
impl OpenAiClient<reqwest::blocking::Client> {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        let base_url = match config.base_url {
            Some(url) => url,
            None => "https://api.openai.com".to_string(),
        };
        Ok(Self {
            api_key: config.api_key,
            base_url,
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
        let url = format!("{}/v1/chat/completions", self.base_url);
        let auth = format!("Bearer {}", self.api_key);

        let resp = self
            .http
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

        Ok((parsed, raw))
    }

    fn send_and_extract(&self, body: Value) -> Result<String, LlmError> {
        let (parsed, raw) = self.send_request(body)?;
        self.extract_text(&parsed, &raw)
    }
}

#[cfg(not(feature = "async"))]
impl LlmClient for OpenAiClient<reqwest::blocking::Client> {
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
        let (parsed, raw) = self.send_request(body)?;
        let token_usage = token_extraction::extract_openai(&parsed);
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
impl OpenAiClient<reqwest::Client> {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        let base_url = match config.base_url {
            Some(url) => url,
            None => "https://api.openai.com".to_string(),
        };
        Ok(Self {
            api_key: config.api_key,
            base_url,
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            thinking: config.thinking,
            response_shape: config.response_shape,
            http: reqwest::Client::builder().timeout(config.timeout).build()?,
        })
    }

    async fn send_request(&self, body: Value) -> Result<(Value, String), LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let auth = format!("Bearer {}", self.api_key);

        let resp = self
            .http
            .post(&url)
            .header("Authorization", auth)
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
impl LlmClient for OpenAiClient<reqwest::Client> {
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
        let prompt = prompt.to_string();
        let attachments = attachments.to_vec();
        let cache_key = cache.and_then(|c| match c {
            CacheResult::Key(id) => Some(id.clone()),
            _ => None,
        });

        Box::pin(async move {
            let prefix = cache_key.as_deref();
            self.send_and_extract(self.build_body(&prompt, &attachments, prefix)?)
                .await
        })
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache: Option<&CacheResult>,
    ) -> BoxFuture<'_, Result<WithTokenUsage<String>, LlmError>> {
        let prompt = prompt.to_string();
        let attachments = attachments.to_vec();
        let cache_key = cache.and_then(|c| match c {
            CacheResult::Key(id) => Some(id.clone()),
            _ => None,
        });

        Box::pin(async move {
            let prefix = cache_key.as_deref();
            let body = self.build_body(&prompt, &attachments, prefix)?;
            let (parsed, raw) = self.send_request(body).await?;
            let token_usage = token_extraction::extract_openai(&parsed);
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
        let content = content.to_string();
        Box::pin(async move { Ok(CacheResult::Key(content)) })
    }
}
