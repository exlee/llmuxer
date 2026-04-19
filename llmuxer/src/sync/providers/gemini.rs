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

pub struct GeminiClient {
    api_key: String,
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    thinking: bool,
    response_shape: ResponseShape,
    client: reqwest::blocking::Client,
}

impl GeminiClient {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        Ok(Self {
            api_key: config.api_key,
            base_url: "https://generativelanguage.googleapis.com".into(),
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

    /// Build parts array for a user turn, including any attachments.
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

    fn send_and_extract(&self, url: String, body: Value) -> Result<String, LlmError> {
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

        parsed["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::Deserialise {
                reason: "could not find text in candidates[0].content.parts[0].text".into(),
                raw,
            })
    }
}

impl LlmClient for GeminiClient {
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

        self.send_and_extract(self.generate_url(), body)
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        cache: Option<&CacheResult>,
    ) -> Result<WithTokenUsage<String>, LlmError> {
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

        let url = self.generate_url();
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

        let parsed: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
                reason: e.to_string(),
                raw: raw.clone(),
            })?;

        let token_usage = token_extraction::extract_gemini(&parsed);
        let result = parsed["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::Deserialise {
                reason: "could not find text in candidates[0].content.parts[0].text".into(),
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
            .client
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
