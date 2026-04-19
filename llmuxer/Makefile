RUSTDOCFLAGS = --cfg docsrs
CHECKS = __check_docs __check_tests __check_tests_async __check_audit __check_docsrs __check_cargo_check __check_clippy __check_fmt


.PHONY: checks
checks: $(CHECKS)
	cargo package --list
	@printf "CHECKS COMPLETE\n"

clean:
	@rm $(CHECKS)

__check_docs:
	$(MAKE) docs
	@touch $@

__check_fmt:
	cargo fmt --check
	@touch $@
__check_tests:
	cargo test
	@touch $@

__check_tests_async:
	cargo test --features async
	@touch $@

__check_audit:
	cd .. && cargo audit
	@touch $@

__check_docsrs:
	cargo +nightly docs-rs
	@touch $@

__check_cargo_check:
	cargo check
	@touch $@

__check_clippy:
	cargo clippy
	@touch $@

.PHONY:docs
docs:
	RUSTDOCFLAGS="$(RUSTDOCFLAGS)" cargo +nightly doc --all-features --no-deps --open
