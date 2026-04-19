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
pub type GeminiProvider = GeminiClient<reqwest::blocking::Client>;
#[cfg(feature = "async")]
pub type GeminiProvider = GeminiClient<reqwest::Client>;

// ---- generic struct + shared logic ----

pub struct GeminiClient<C> {
    api_key: String,
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    thinking: bool,
    response_shape: ResponseShape,
    http: C,
}

impl<C> GeminiClient<C> {
    fn generate_url(&self) -> String {
        format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        )
    }

    fn build_generation_config(&self) -> Value {
        let mut config = json!({"maxOutputTokens": self.max_tokens});

        if self.thinking {
            config["thinkingConfig"] = json!({"thinkingBudget": -1});
        }

        match &self.response_shape {
            ResponseShape::Text => {}
            ResponseShape::Json(schema) => {
                config["responseMimeType"] = json!("application/json");
                config["responseSchema"] = schema.clone();
            }
        }

        config
    }

    fn build_parts(prompt: &str, attachments: &[Attachment]) -> Result<Vec<Value>, LlmError> {
        let mut parts = Vec::new();

        for att in attachments {
            let (bytes, mime) = att.resolve()?;
            parts.push(json!({
                "inlineData": {
                    "mimeType": mime,
                    "data": B64.encode(&bytes)
                }
            }));
        }

        parts.push(json!({"text": prompt}));
        Ok(parts)
    }

    fn build_body(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache: Option<&CacheResult>,
    ) -> Result<Value, LlmError> {
        let parts = Self::build_parts(prompt, attachments)?;

        let body = match cache {
            Some(CacheResult::Key(id)) => json!({
                "cachedContent": id,
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": self.build_generation_config()
            }),
            _ => json!({
                "systemInstruction": {"parts": [{"text": self.instruction}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": self.build_generation_config()
            }),
        };

        Ok(body)
    }

    fn extract_text(&self, parsed: &Value, raw: &str) -> Result<String, LlmError> {
        parsed["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::Deserialise {
                reason: "could not find text in candidates[0].content.parts[0].text".into(),
                raw: raw.to_string(),
            })
    }
}

// ---- sync implementation ----

#[cfg(not(feature = "async"))]
use crate::traits::LlmClient;

#[cfg(not(feature = "async"))]
impl GeminiClient<reqwest::blocking::Client> {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        Ok(Self {
            api_key: config.api_key,
            base_url: "https://generativelanguage.googleapis.com".into(),
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

    fn send_request(&self, url: String, body: Value) -> Result<(Value, String), LlmError> {
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

    fn send_and_extract(&self, url: String, body: Value) -> Result<String, LlmError> {
        let (parsed, raw) = self.send_request(url, body)?;
        self.extract_text(&parsed, &raw)
    }
}

#[cfg(not(feature = "async"))]
impl LlmClient for GeminiClient<reqwest::blocking::Client> {
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
        let body = self.build_body(prompt, attachments, cache)?;
        self.send_and_extract(self.generate_url(), body)
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache: Option<&CacheResult>,
    ) -> Result<WithTokenUsage<String>, LlmError> {
        let body = self.build_body(prompt, attachments, cache)?;
        let url = self.generate_url();
        let (parsed, raw) = self.send_request(url, body)?;
        let token_usage = token_extraction::extract_gemini(&parsed);
        let result = self.extract_text(&parsed, &raw)?;
        Ok(WithTokenUsage {
            token_usage,
            result,
        })
    }

    fn execute_cache(
        &self,
        content: &str,
        attachments: &[Attachment],
    ) -> Result<CacheResult, LlmError> {
        let url = format!(
            "{}/v1beta/cachedContents?key={}",
            self.base_url, self.api_key
        );

        let mut parts: Vec<Value> = Vec::new();
        for att in attachments {
            let (bytes, mime) = att.resolve()?;
            parts.push(json!({
                "inlineData": {
                    "mimeType": mime,
                    "data": B64.encode(&bytes)
                }
            }));
        }
        parts.push(json!({"text": content}));

        let body = json!({
            "model": format!("models/{}", self.model),
            "contents": [{"role": "user", "parts": parts}],
            "ttl": "3600s"
        });

        let resp = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(LlmError::Request)?;

        let status = resp.status().as_u16();
        let raw = resp.text().map_err(LlmError::Request)?;

        if status >= 400 {
            return Err(LlmError::ProviderError { status, body: raw });
        }

        let parsed: Value = serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
            reason: e.to_string(),
            raw: raw.clone(),
        })?;

        parsed["name"]
            .as_str()
            .map(|s| CacheResult::Key(s.to_string()))
            .ok_or_else(|| LlmError::Cache("no name in cachedContents response".into()))
    }
}

// ---- async implementation ----

#[cfg(feature = "async")]
use crate::traits::LlmClient;
#[cfg(feature = "async")]
use futures::future::BoxFuture;

#[cfg(feature = "async")]
impl GeminiClient<reqwest::Client> {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        Ok(Self {
            api_key: config.api_key,
            base_url: "https://generativelanguage.googleapis.com".into(),
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            thinking: config.thinking,
            response_shape: config.response_shape,
            http: reqwest::Client::builder().timeout(config.timeout).build()?,
        })
    }

    async fn send_request(&self, url: String, body: Value) -> Result<(Value, String), LlmError> {
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

    async fn send_and_extract(&self, url: String, body: Value) -> Result<String, LlmError> {
        let (parsed, raw) = self.send_request(url, body).await?;
        self.extract_text(&parsed, &raw)
    }
}

#[cfg(feature = "async")]
impl LlmClient for GeminiClient<reqwest::Client> {
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
        let cache = cache.cloned();

        Box::pin(async move {
            let body = self.build_body(&prompt, &attachments, cache.as_ref())?;
            self.send_and_extract(self.generate_url(), body).await
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
        let cache = cache.cloned();

        Box::pin(async move {
            let body = self.build_body(&prompt, &attachments, cache.as_ref())?;
            let url = self.generate_url();
            let (parsed, raw) = self.send_request(url, body).await?;
            let token_usage = token_extraction::extract_gemini(&parsed);
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
        attachments: &[Attachment],
    ) -> BoxFuture<'_, Result<CacheResult, LlmError>> {
        let content = content.to_string();
        let attachments = attachments.to_vec();

        Box::pin(async move {
            let url = format!(
                "{}/v1beta/cachedContents?key={}",
                self.base_url, self.api_key
            );

            let mut parts: Vec<Value> = Vec::new();
            for att in &attachments {
                let (bytes, mime) = att.resolve()?;
                parts.push(json!({
                    "inlineData": {
                        "mimeType": mime,
                        "data": B64.encode(&bytes)
                    }
                }));
            }
            parts.push(json!({"text": content}));

            let body = json!({
                "model": format!("models/{}", self.model),
                "contents": [{"role": "user", "parts": parts}],
                "ttl": "3600s"
            });

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

            parsed["name"]
                .as_str()
                .map(|s| CacheResult::Key(s.to_string()))
                .ok_or_else(|| LlmError::Cache("no name in cachedContents response".into()))
        })
    }
}
