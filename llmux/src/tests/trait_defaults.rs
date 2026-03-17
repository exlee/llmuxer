#[cfg(test)]
mod tests {
    use crate::{CacheResult, LlmClient, LlmClientExt, LlmError};

    struct AlwaysOk(String);

    impl LlmClient for AlwaysOk {
        fn query(&self, _: &str) -> Result<String, LlmError> {
            Ok(self.0.clone())
        }
    }

    struct AlwaysOkWithCache;

    impl LlmClient for AlwaysOkWithCache {
        fn query(&self, _: &str) -> Result<String, LlmError> {
            Ok("fallback".into())
        }
        fn query_with_cache(&self, _: &str, cache_id: &str) -> Result<String, LlmError> {
            Ok(format!("cached:{cache_id}"))
        }
    }

    #[test]
    fn query_cached_falls_back_on_unsupported() {
        let client = AlwaysOk("pong".into());
        let result = client
            .query_cached("ping", &CacheResult::Unsupported)
            .unwrap();
        assert_eq!(result, "pong");
    }

    #[test]
    fn query_cached_falls_back_on_err() {
        let client = AlwaysOk("pong".into());
        let cache = CacheResult::Err(LlmError::Cache("boom".into()));
        let result = client.query_cached("ping", &cache).unwrap();
        assert_eq!(result, "pong");
    }

    #[test]
    fn query_cached_uses_cache_key_when_present() {
        let client = AlwaysOkWithCache;
        let cache = CacheResult::Key("abc123".into());
        let result = client.query_cached("ping", &cache).unwrap();
        assert_eq!(result, "cached:abc123");
    }

    #[test]
    fn query_json_deserialises_valid_response() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Foo {
            x: u32,
        }

        let client = AlwaysOk(r#"{"x": 42}"#.into());
        let result: Foo = client.query_json("anything").unwrap();
        assert_eq!(result, Foo { x: 42 });
    }

    #[test]
    fn query_json_returns_deserialise_error_with_raw() {
        #[derive(serde::Deserialize)]
        struct Foo {
            #[allow(dead_code)]
            x: u32,
        }

        let client = AlwaysOk("not json".into());
        let err = client.query_json::<Foo>("anything").err().unwrap();
        assert!(matches!(err, LlmError::Deserialise { raw, .. } if raw == "not json"));
    }

    #[test]
    fn build_cache_returns_unsupported_by_default() {
        let client = AlwaysOk("x".into());
        let result = client.build_cache("content");
        assert!(matches!(result, CacheResult::Unsupported));
    }
}
