#[cfg(test)]
mod tests {
    use crate::attachment::Attachment;
    use crate::traits::{CacheBuilder, QueryBuilder};
    use crate::{CacheResult, LlmClient, LlmError};

    struct AlwaysOk(String);

    impl LlmClient for AlwaysOk {
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
            _prompt: &str,
            _attachments: &[Attachment],
            _cache: Option<&CacheResult>,
        ) -> Result<String, LlmError> {
            Ok(self.0.clone())
        }

        fn execute_cache(
            &self,
            _content: &str,
            _attachments: &[Attachment],
        ) -> Result<CacheResult, LlmError> {
            Ok(CacheResult::Unsupported)
        }
    }

    struct CacheAware;

    impl LlmClient for CacheAware {
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
            _prompt: &str,
            _attachments: &[Attachment],
            cache: Option<&CacheResult>,
        ) -> Result<String, LlmError> {
            match cache {
                Some(CacheResult::Key(id)) => Ok(format!("cached:{id}")),
                _ => Ok("fallback".into()),
            }
        }

        fn execute_cache(
            &self,
            content: &str,
            _attachments: &[Attachment],
        ) -> Result<CacheResult, LlmError> {
            Ok(CacheResult::Key(content.to_string()))
        }
    }

    #[test]
    fn run_without_cache_returns_response() {
        let client = AlwaysOk("pong".into());
        let result = client.query("ping").run().unwrap();
        assert_eq!(result, "pong");
    }

    #[test]
    fn run_with_unsupported_cache_falls_back() {
        let client = CacheAware;
        let cache = CacheResult::Unsupported;
        let result = client.query("ping").cache(&cache).run().unwrap();
        assert_eq!(result, "fallback");
    }

    #[test]
    fn run_with_cache_key_uses_cache() {
        let client = CacheAware;
        let cache = CacheResult::Key("abc123".into());
        let result = client.query("ping").cache(&cache).run().unwrap();
        assert_eq!(result, "cached:abc123");
    }

    #[test]
    fn require_cache_errors_when_no_cache_provided() {
        let client = CacheAware;
        let err = client.query("ping").require_cache().run().unwrap_err();
        assert!(matches!(err, LlmError::CacheRequired));
    }

    #[test]
    fn require_cache_errors_when_cache_is_unsupported() {
        let client = CacheAware;
        let cache = CacheResult::Unsupported;
        let err = client
            .query("ping")
            .cache(&cache)
            .require_cache()
            .run()
            .unwrap_err();
        assert!(matches!(err, LlmError::CacheRequired));
    }

    #[test]
    fn require_cache_succeeds_with_key() {
        let client = CacheAware;
        let cache = CacheResult::Key("key".into());
        let result = client
            .query("ping")
            .cache(&cache)
            .require_cache()
            .run()
            .unwrap();
        assert_eq!(result, "cached:key");
    }

    #[test]
    fn json_deserialises_valid_response() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Foo {
            x: u32,
        }

        let client = AlwaysOk(r#"{"x": 42}"#.into());
        let result: Foo = client.query("anything").json().unwrap();
        assert_eq!(result, Foo { x: 42 });
    }

    #[test]
    fn json_returns_deserialise_error_with_raw() {
        #[derive(Debug, serde::Deserialize)]
        struct Foo {
            #[allow(dead_code)]
            x: u32,
        }

        let client = AlwaysOk("not json".into());
        let err = client.query("anything").json::<Foo>().unwrap_err();
        assert!(matches!(err, LlmError::Deserialise { raw, .. } if raw == "not json"));
    }

    #[test]
    fn build_cache_returns_unsupported_for_default_impl() {
        let client = AlwaysOk("x".into());
        let result = client.build_cache("content").build().unwrap();
        assert!(matches!(result, CacheResult::Unsupported));
    }

    #[test]
    fn build_cache_returns_key_when_supported() {
        let client = CacheAware;
        let result = client.build_cache("ctx").build().unwrap();
        assert!(matches!(result, CacheResult::Key(k) if k == "ctx"));
    }
}
