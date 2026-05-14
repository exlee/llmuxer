# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-05-14

### Added

- `Provider::LlamaCpp` — llama.cpp server provider using the OAI-compatible
  `/v1/chat/completions` endpoint (no API key, default `http://127.0.0.1:8080`)
- `thinking_budget` option on `LlmClientBuilder` — Anthropic maps to
  `budget_tokens`, Gemini maps to `thinkingBudget`, others store but do not use
- `ReasoningEffort` enum (`Low`, `Medium`, `High`) on `LlmClientBuilder` —
  OpenAI maps to `reasoning_effort`, OpenRouter maps to `reasoning.effort`,
  others store but do not use
- `extract_llamacpp` token extraction — reads `timings.prompt_n`,
  `timings.cache_n`, `timings.predicted_n` with fallback to standard `usage`
- LlamaCpp provider in `llmuxer-egui` provider picker with base URL field
- OpenRouter as a standalone provider

### Changed

- Detached `CacheResult` and `WithTokenUsage` from client lifetime for
  improved ergonomics
- Added async `execute_query_with_tokens` / `execute_cache` trait methods
- Anthropic: structured system message with `cache_control: ephemeral` when
  cache key is present
- OpenAI/OpenRouter: system_prefix threaded into system message for cache
  separation

## [0.2.0]

### Added

- `async` feature (without it `reqwest::blocking` is used by default)
- Results with token count through `QueryBuilder.with_tokens` method

### Changed

- `reqwest` uses `rustls-tls` now, so no openssl compilation
- `OpenAiProvider` accepts `base_url` configuration for OpenAI-compatible models (e.g. OpenRouter)

[0.3.0]: https://github.com/example/llmuxer/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/example/llmuxer/releases/tag/v0.2.0
