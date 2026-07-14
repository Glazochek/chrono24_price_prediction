.PHONY: help setup lint format test data train docker clean
.DEFAULT_GOAL := help

help: ## Show the available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

setup: ## Install everything from the lockfile
	uv sync

lint: ## Check style and formatting (Ruff)
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix and format the code
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite
	uv run pytest -q

data: ## Download the dataset and print its shape (needs Kaggle creds in .env)
	uv run python -c "from chrono24 import load_watches; print(load_watches().shape)"

train: ## Run the full pipeline: clean, remove outliers, train, save model + plots
	uv run python -m chrono24

docker: ## Build the container image
	docker build -t chrono24 .

clean: ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
