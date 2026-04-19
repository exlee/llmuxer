#[cfg(test)]
mod tests {
    use crate::LlmError;
    use crate::attachment::Attachment;
    use crate::traits::{CacheBuilder, CacheResult, LlmClient, QueryBuilder};

    #[cfg(feature = "async")]
    use futures::future::BoxFuture;

    // ── helpers to abstract over sync / async ──────────────────────────

    #[cfg(not(feature = "async"))]
    fn run_query(qb: QueryBuilder<'_>) -> Result<String, LlmError> {
        qb.run()
    }

    #[cfg(feature = "async")]
    fn run_query(qb: QueryBuilder<'_>) -> Result<String, LlmError> {
        futures::executor::block_on(qb.run())
    }

    #[cfg(not(feature = "async"))]
    fn run_json<T: serde::de::DeserializeOwned>(qb: QueryBuilder<'_>) -> Result<T, LlmError> {
        qb.json()
    }

    #[cfg(feature = "async")]
    fn run_json<T: serde::de::DeserializeOwned>(qb: QueryBuilder<'_>) -> Result<T, LlmError> {
        futures::executor::block_on(qb.json())
    }

    #[cfg(not(feature = "async"))]
    fn run_cache_build(cb: CacheBuilder<'_>) -> Result<CacheResult, LlmError> {
        cb.build()
    }

    #[cfg(feature = "async")]
    fn run_cache_build(cb: CacheBuilder<'_>) -> Result<CacheResult, LlmError> {
        futures::executor::block_on(cb.build())
    }

    // ── mock: AlwaysOk ─────────────────────────────────────────────────

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

        #[cfg(not(feature = "async"))]
        fn execute_query(
            &self,
            _prompt: &str,
            _attachments: &[Attachment],
            _cache: Option<&CacheResult>,
        ) -> Result<String, LlmError> {
            Ok(self.0.clone())
        }

        #[cfg(feature = "async")]
        fn execute_query(
            &self,
            _prompt: &str,
            _attachments: &[Attachment],
            _cache: Option<&CacheResult>,
        ) -> BoxFuture<'_, Result<String, LlmError>> {
            let result = self.0.clone();
            Box::pin(async move { Ok(result) })
        }

        #[cfg(not(feature = "async"))]
        fn execute_cache(
            &self,
            _content: &str,
            _attachments: &[Attachment],
        ) -> Result<CacheResult, LlmError> {
            Ok(CacheResult::Unsupported)
        }

        #[cfg(feature = "async")]
        fn execute_cache(
            &self,
            _content: &str,
            _attachments: &[Attachment],
        ) -> BoxFuture<'_, Result<CacheResult, LlmError>> {
            Box::pin(async move { Ok(CacheResult::Unsupported) })
        }
    }

    // ── mock: CacheAware ───────────────────────────────────────────────

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

        #[cfg(not(feature = "async"))]
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

        #[cfg(feature = "async")]
        fn execute_query(
            &self,
            _prompt: &str,
            _attachments: &[Attachment],
            cache: Option<&CacheResult>,
        ) -> BoxFuture<'_, Result<String, LlmError>> {
            let result = match cache {
                Some(CacheResult::Key(id)) => format!("cached:{id}"),
                _ => "fallback".into(),
            };
            Box::pin(async move { Ok(result) })
        }

        #[cfg(not(feature = "async"))]
        fn execute_cache(
            &self,
            content: &str,
            _attachments: &[Attachment],
        ) -> Result<CacheResult, LlmError> {
            Ok(CacheResult::Key(content.to_string()))
        }

        #[cfg(feature = "async")]
        fn execute_cache(
            &self,
            content: &str,
            _attachments: &[Attachment],
        ) -> BoxFuture<'_, Result<CacheResult, LlmError>> {
            let content = content.to_owned();
            Box::pin(async move { Ok(CacheResult::Key(content)) })
        }
    }

    // ── tests ──────────────────────────────────────────────────────────

    #[test]
    fn run_without_cache_returns_response() {
        let client = AlwaysOk("pong".into());
        let result = run_query(client.query("ping")).unwrap();
        assert_eq!(result, "pong");
    }

    #[test]
    fn run_with_unsupported_cache_falls_back() {
        let client = CacheAware;
        let cache = CacheResult::Unsupported;
        let result = run_query(client.query("ping").cache(&cache)).unwrap();
        assert_eq!(result, "fallback");
    }

    #[test]
    fn run_with_cache_key_uses_cache() {
        let client = CacheAware;
        let cache = CacheResult::Key("abc123".into());
        let result = run_query(client.query("ping").cache(&cache)).unwrap();
        assert_eq!(result, "cached:abc123");
    }

    #[test]
    fn require_cache_errors_when_no_cache_provided() {
        let client = CacheAware;
        let err = run_query(client.query("ping").require_cache()).unwrap_err();
        assert!(matches!(err, LlmError::CacheRequired));
    }

    #[test]
    fn require_cache_errors_when_cache_is_unsupported() {
        let client = CacheAware;
        let cache = CacheResult::Unsupported;
        let err = run_query(client.query("ping").cache(&cache).require_cache()).unwrap_err();
        assert!(matches!(err, LlmError::CacheRequired));
    }

    #[test]
    fn require_cache_succeeds_with_key() {
        let client = CacheAware;
        let cache = CacheResult::Key("key".into());
        let result = run_query(client.query("ping").cache(&cache).require_cache()).unwrap();
        assert_eq!(result, "cached:key");
    }

    #[test]
    fn json_deserialises_valid_response() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Foo {
            x: u32,
        }

        let client = AlwaysOk(r#"{"x": 42}"#.into());
        let result: Foo = run_json(client.query("anything")).unwrap();
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
        let err = run_json::<Foo>(client.query("anything")).unwrap_err();
        assert!(matches!(err, LlmError::Deserialise { ref raw, .. } if raw == "not json"));
    }

    #[test]
    fn build_cache_returns_unsupported_for_default_impl() {
        let client = AlwaysOk("x".into());
        let result = run_cache_build(client.build_cache("content")).unwrap();
        assert!(matches!(result, CacheResult::Unsupported));
    }

    #[test]
    fn build_cache_returns_key_when_supported() {
        let client = CacheAware;
        let result = run_cache_build(client.build_cache("ctx")).unwrap();
        assert!(matches!(result, CacheResult::Key(ref k) if k == "ctx"));
    }
}
