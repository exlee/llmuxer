# llmux

Core crate. Provides `LlmClient`, `LlmClientBuilder`, and provider implementations for Anthropic, Gemini, OpenAI, and Ollama.

## Usage

```rust
use llmux::{LlmClientBuilder, Provider, ResponseShape};
use serde_json::json;

// Plain text query
let client = LlmClientBuilder::new()
    .provider(Provider::OpenAI)
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .instruction("You are a concise assistant.")
    .max_tokens(512)
    .build()?;

let text = client.query("Summarise the Rust ownership model in one sentence.")?;

// Structured JSON output
use llmux::LlmClientExt;

#[derive(serde::Deserialize)]
struct Summary { sentence: String }

let schema = json!({
    "type": "object",
    "properties": { "sentence": { "type": "string" } },
    "required": ["sentence"]
});

let client = LlmClientBuilder::new()
    .provider(Provider::Anthropic)
    .api_key(std::env::var("ANTHROPIC_API_KEY")?)
    .response_shape(ResponseShape::Json(schema))
    .build()?;

let result: Summary = client.query_json("Summarise Rust ownership.")?;
```

## Context caching

```rust
let client = LlmClientBuilder::new()
    .provider(Provider::Anthropic)
    .api_key(std::env::var("ANTHROPIC_API_KEY")?)
    .build()?;

let large_doc = std::fs::read_to_string("large_document.txt")?;
let cache = client.build_cache(&large_doc);

let answer = client.query_cached("What is the main topic?", &cache)?;
```

## Providers

| Provider | Caching | Thinking | Needs key |
|---|---|---|---|
| Anthropic | Yes | Yes | Yes |
| Gemini | Yes | Yes | Yes |
| OpenAI | Yes (automatic) | Yes | Yes |
| Ollama | No | No | No (needs `base_url`) |
