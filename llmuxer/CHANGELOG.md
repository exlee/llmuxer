# Changelog

## 0.2.0

### Added

- `async` feature (without it `reqwest::blocking` is used by default)
- Results with token count through `QueryBuilder.with_tokens` method

### Changed

- `reqwest` uses `rustls-tls` now, so no openssl compilation
- `OpenAiProvider` accepts `base_url` configuration for OpenAI-compatible models (e.g. OpenRouter)
