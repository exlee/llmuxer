use serde_json::Value;

use crate::token_usage::TokenUsage;

/// Extract token usage from an Anthropic API response.
///
/// Anthropic returns a top-level `usage` object:
/// ```json
/// { "usage": { "input_tokens": 25, "output_tokens": 150,
///              "cache_read_input_tokens": 0 } }
/// ```
pub fn extract_anthropic(parsed: &Value) -> TokenUsage {
    let usage = &parsed["usage"];
    TokenUsage {
        prompt_token_count: usage["input_tokens"].as_u64().map(|n| n as usize),
        cached_content_token_count: usage["cache_read_input_tokens"]
            .as_u64()
            .map(|n| n as usize),
        thoughts_token_count: None,
        output_token_count: usage["output_tokens"].as_u64().map(|n| n as usize),
        total_token_count: None,
    }
}

/// Extract token usage from an OpenAI API response.
///
/// OpenAI returns a top-level `usage` object:
/// ```json
/// { "usage": {
///     "prompt_tokens": 25, "completion_tokens": 150, "total_tokens": 175,
///     "prompt_tokens_details": { "cached_tokens": 0 },
///     "completion_tokens_details": { "reasoning_tokens": 0 }
/// } }
/// ```
pub fn extract_openai(parsed: &Value) -> TokenUsage {
    let usage = &parsed["usage"];
    TokenUsage {
        prompt_token_count: usage["prompt_tokens"].as_u64().map(|n| n as usize),
        cached_content_token_count: usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .map(|n| n as usize),
        thoughts_token_count: usage["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .map(|n| n as usize),
        output_token_count: usage["completion_tokens"].as_u64().map(|n| n as usize),
        total_token_count: usage["total_tokens"].as_u64().map(|n| n as usize),
    }
}

/// Extract token usage from a Gemini API response.
///
/// Gemini returns a top-level `usageMetadata` object:
/// ```json
/// { "usageMetadata": {
///     "promptTokenCount": 25, "candidatesTokenCount": 150,
///     "totalTokenCount": 175, "cachedContentTokenCount": 0,
///     "thoughtsTokenCount": 0
/// } }
/// ```
pub fn extract_gemini(parsed: &Value) -> TokenUsage {
    let meta = &parsed["usageMetadata"];
    TokenUsage {
        prompt_token_count: meta["promptTokenCount"].as_u64().map(|n| n as usize),
        cached_content_token_count: meta["cachedContentTokenCount"].as_u64().map(|n| n as usize),
        thoughts_token_count: meta["thoughtsTokenCount"].as_u64().map(|n| n as usize),
        output_token_count: meta["candidatesTokenCount"].as_u64().map(|n| n as usize),
        total_token_count: meta["totalTokenCount"].as_u64().map(|n| n as usize),
    }
}

/// Extract token usage from an OpenRouter API response.
///
/// OpenRouter uses the same response shape as OpenAI:
/// ```json
/// { "usage": {
///     "prompt_tokens": 25, "completion_tokens": 150, "total_tokens": 175,
///     "prompt_tokens_details": { "cached_tokens": 0 },
///     "completion_tokens_details": { "reasoning_tokens": 0 }
/// } }
/// ```
pub fn extract_openrouter(parsed: &Value) -> TokenUsage {
    extract_openai(parsed)
}

/// Extract token usage from an Ollama API response.
///
/// Ollama returns top-level counters:
/// ```json
/// { "prompt_eval_count": 25, "eval_count": 150 }
/// ```
pub fn extract_ollama(parsed: &Value) -> TokenUsage {
    TokenUsage {
        prompt_token_count: parsed["prompt_eval_count"].as_u64().map(|n| n as usize),
        cached_content_token_count: None,
        thoughts_token_count: None,
        output_token_count: parsed["eval_count"].as_u64().map(|n| n as usize),
        total_token_count: None,
    }
}

/// Extract token usage from a llama.cpp server OAI-compatible response.
///
/// The llama.cpp server returns a `timings` object alongside the OAI-standard
/// `usage` object when queried via `/v1/chat/completions`:
/// ```json
/// {
///   "usage": { "prompt_tokens": 25, "completion_tokens": 150, "total_tokens": 175 },
///   "timings": {
///     "prompt_n": 25, "cache_n": 10,
///     "predicted_n": 150,
///     "prompt_ms": 30.0, "predicted_ms": 600.0
///   }
/// }
/// ```
///
/// We prefer the `timings` fields (they include cache info) and fall back to
/// the standard `usage` object.
pub fn extract_llamacpp(parsed: &Value) -> TokenUsage {
    let timings = &parsed["timings"];

    let prompt_token_count = timings["prompt_n"]
        .as_u64()
        .or_else(|| parsed["usage"]["prompt_tokens"].as_u64())
        .map(|n| n as usize);

    // cache_n = tokens reused from KV cache
    let cached_content_token_count = timings["cache_n"].as_u64().map(|n| n as usize);

    let output_token_count = timings["predicted_n"]
        .as_u64()
        .or_else(|| parsed["usage"]["completion_tokens"].as_u64())
        .map(|n| n as usize);

    let total_token_count = parsed["usage"]["total_tokens"]
        .as_u64()
        .map(|n| n as usize);

    TokenUsage {
        prompt_token_count,
        cached_content_token_count,
        thoughts_token_count: None,
        output_token_count,
        total_token_count,
    }
}
