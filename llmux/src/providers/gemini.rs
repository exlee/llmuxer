use serde_json::{json, Value};

use crate::{
    builder::{ClientConfig, ResponseShape},
    error::LlmError,
    traits::{CacheResult, LlmClient},
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
            client: reqwest::blocking::Client::new(),
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

    fn build_body(&self, query: &str) -> Value {
        json!({
            "systemInstruction": {"parts": [{"text": self.instruction}]},
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            "generationConfig": self.build_generation_config()
        })
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

        let parsed: Value =
            serde_json::from_str(&raw).map_err(|e| LlmError::Deserialise {
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
    fn query(&self, query: &str) -> Result<String, LlmError> {
        self.send_and_extract(self.generate_url(), self.build_body(query))
    }

    fn query_with_cache(&self, query: &str, cache_id: &str) -> Result<String, LlmError> {
        let body = json!({
            "cachedContent": cache_id,
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            "generationConfig": self.build_generation_config()
        });
        self.send_and_extract(self.generate_url(), body)
    }

    fn build_cache(&self, content: &str) -> CacheResult {
        let url = format!("{}/v1beta/cachedContents?key={}", self.base_url, self.api_key);
        let body = json!({
            "model": format!("models/{}", self.model),
            "contents": [{"role": "user", "parts": [{"text": content}]}],
            "ttl": "3600s"
        });

        let result = (|| -> Result<String, LlmError> {
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
                .map(|s| s.to_string())
                .ok_or_else(|| LlmError::Cache("no name in cachedContents response".into()))
        })();

        match result {
            Ok(name) => CacheResult::Key(name),
            Err(e) => CacheResult::Err(e),
        }
    }
}
