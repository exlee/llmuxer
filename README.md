# llmuxer

Synchronous Rust client library for Anthropic, Gemini, OpenAI, and Ollama with a unified trait interface.

## Workspace crates

| Crate | Description |
|---|---|
| [`llmuxer`](llmuxer/) | Core client trait and provider implementations |
| [`llmuxer-keystore`](llmuxer-keystore/) | Credential persistence to `~/.config/llmuxer/config.json` |
| [`llmuxer-egui`](llmuxer-egui/) | egui widget for configuring providers at runtime |

## Quick start

```rust
use llmuxer::{LlmClientBuilder, Provider};

let client = LlmClientBuilder::new()
    .provider(Provider::Anthropic)
    .api_key("sk-ant-...")
    .instruction("You are a helpful assistant.")
    .build()?;

let response = client.query("What is 2 + 2?")?;
```

## Features

- Single `LlmClient` trait works across all providers
- Structured JSON output via `ResponseShape::Json(schema)`
- Context caching for Anthropic and Gemini
- Extended thinking for Anthropic, Gemini, and OpenAI
- Local models via Ollama

## License

MIT
