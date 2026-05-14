use serde_json::{Value, json};

use crate::{
    attachment::Attachment,
    builder::{ClientConfig, ReasoningEffort, ResponseShape},
    error::LlmError,
    shared::CacheResult,
    token_extraction,
    token_usage::WithTokenUsage,
};

// ---- public type alias for the active mode ----

#[cfg(not(feature = "async"))]
pub type LlamaCppProvider = LlamaCppClient<reqwest::blocking::Client>;
#[cfg(feature = "async")]
pub type LlamaCppProvider = LlamaCppClient<reqwest::Client>;

// ---- generic struct + shared logic ----

pub struct LlamaCppClient<C> {
    base_url: String,
    model: String,
    instruction: String,
    max_tokens: u32,
    thinking: bool,
    #[allow(dead_code)]
    thinking_budget: Option<u32>,
    #[allow(dead_code)]
    reasoning_effort: ReasoningEffort,
    response_shape: ResponseShape,
    http: C,
}

impl<C> LlamaCppClient<C> {
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

    fn build_body(&self, prompt: &str, _attachments: &[Attachment]) -> Result<Value, LlmError> {
        // llama.cpp server uses the OAI-compatible /v1/chat/completions endpoint.
        // It supports json_schema response_format natively.
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.build_system()},
                {"role": "user", "content": prompt}
            ]
        });

        // When thinking is disabled, explicitly turn off reasoning parsing.
        // When enabled, omit the field so the server uses its default (auto-detect
        // from the model's chat template), which is the desired behaviour.
        if !self.thinking {
            body["reasoning_format"] = json!("none");
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
impl LlamaCppClient<reqwest::blocking::Client> {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        let base_url = config
            .base_url
            .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());
        Ok(Self {
            base_url,
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            thinking: config.thinking,
            thinking_budget: config.thinking_budget,
            reasoning_effort: config.reasoning_effort,
            response_shape: config.response_shape,
            http: reqwest::blocking::Client::builder()
                .timeout(config.timeout)
                .build()?,
        })
    }

    fn send_request(&self, body: Value) -> Result<(Value, String), LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let req = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .json(&body);

        let resp = req.send()?;

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
impl LlmClient for LlamaCppClient<reqwest::blocking::Client> {
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
        _cache: Option<CacheResult>,
    ) -> Result<String, LlmError> {
        self.send_and_extract(self.build_body(prompt, attachments)?)
    }

    fn execute_query_with_tokens(
        &self,
        prompt: &str,
        attachments: &[Attachment],
        _cache: Option<CacheResult>,
    ) -> Result<WithTokenUsage<String>, LlmError> {
        let body = self.build_body(prompt, attachments)?;
        let (parsed, raw) = self.send_request(body)?;
        let token_usage = token_extraction::extract_llamacpp(&parsed);
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
impl LlamaCppClient<reqwest::Client> {
    pub(crate) fn new(config: ClientConfig) -> Result<Self, LlmError> {
        let base_url = config
            .base_url
            .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());
        Ok(Self {
            base_url,
            model: config.model,
            instruction: config.instruction,
            max_tokens: config.max_tokens,
            thinking: config.thinking,
            thinking_budget: config.thinking_budget,
            reasoning_effort: config.reasoning_effort,
            response_shape: config.response_shape,
            http: reqwest::Client::builder().timeout(config.timeout).build()?,
        })
    }

    async fn send_request(&self, body: Value) -> Result<(Value, String), LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

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
impl LlmClient for LlamaCppClient<reqwest::Client> {
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
        _cache: Option<CacheResult>,
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
        _cache: Option<CacheResult>,
    ) -> BoxFuture<'_, Result<WithTokenUsage<String>, LlmError>> {
        let prompt = prompt.to_string();
        let attachments = attachments.to_vec();

        Box::pin(async move {
            let body = self.build_body(&prompt, &attachments)?;
            let (parsed, raw) = self.send_request(body).await?;
            let token_usage = token_extraction::extract_llamacpp(&parsed);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ReasoningEffort;

    fn make_client(thinking: bool, shape: ResponseShape) -> LlamaCppClient<()> {
        LlamaCppClient {
            base_url: "http://127.0.0.1:8080".into(),
            model: "test-model".into(),
            instruction: "You are helpful.".into(),
            max_tokens: 1024,
            thinking,
            thinking_budget: None,
            reasoning_effort: ReasoningEffort::default(),
            response_shape: shape,
            http: (),
        }
    }

    #[test]
    fn thinking_off_sets_reasoning_format_none() {
        let client = make_client(false, ResponseShape::Text);
        let body = client.build_body("hello", &[]).unwrap();
        assert_eq!(body["reasoning_format"], "none");
    }

    #[test]
    fn thinking_on_omits_reasoning_format() {
        let client = make_client(true, ResponseShape::Text);
        let body = client.build_body("hello", &[]).unwrap();
        assert!(body.get("reasoning_format").is_none());
    }

    #[test]
    fn json_schema_sets_response_format() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}}
        });
        let client = make_client(false, ResponseShape::Json(schema.clone()));
        let body = client.build_body("hello", &[]).unwrap();

        let rf = &body["response_format"];
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "response");
        assert_eq!(rf["json_schema"]["strict"], true);
        assert_eq!(rf["json_schema"]["schema"], schema);
    }

    #[test]
    fn text_shape_has_no_response_format() {
        let client = make_client(false, ResponseShape::Text);
        let body = client.build_body("hello", &[]).unwrap();
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn system_prompt_includes_schema_when_json_shape() {
        let schema = serde_json::json!({"type": "array", "items": {"type": "number"}});
        let client = make_client(false, ResponseShape::Json(schema));
        let body = client.build_body("hello", &[]).unwrap();
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(sys.contains("Respond with JSON matching this schema"));
        assert!(sys.contains("You are helpful."));
    }

    #[test]
    fn system_prompt_is_plain_instruction_for_text_shape() {
        let client = make_client(false, ResponseShape::Text);
        let body = client.build_body("hello", &[]).unwrap();
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert_eq!(sys, "You are helpful.");
    }

    #[test]
    fn extract_text_from_valid_response() {
        let client = make_client(false, ResponseShape::Text);
        let parsed = serde_json::json!({
            "choices": [{ "message": { "content": "hello world" } }]
        });
        assert_eq!(client.extract_text(&parsed, "").unwrap(), "hello world");
    }

    #[test]
    fn extract_text_returns_deserialise_error_with_raw() {
        let client = make_client(false, ResponseShape::Text);
        let parsed = serde_json::json!({ "choices": [{ "message": {} }] });
        let raw = "{\"choices\":[]}";
        let err = client.extract_text(&parsed, raw).unwrap_err();
        match err {
            LlmError::Deserialise { raw: r, .. } => assert_eq!(r, "{\"choices\":[]}"),
            other => panic!("expected Deserialise, got {other:?}"),
        }
    }
}
