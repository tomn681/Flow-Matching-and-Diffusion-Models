.PHONY: test typecheck-public

test:
	./.venv/bin/pytest -q

typecheck-public:
	./scripts/typecheck_public_api.sh
