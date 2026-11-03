# Convenience targets (Linux/macOS/Git-Bash). Windows PowerShell users: run the python commands directly.
UV := uv run --project backend

.PHONY: demo prepare test test-unit test-integration lint frontend bench faults soak down

demo:             ## one-command start (prereqs, .env, compose, deps, clips, weights, UI, app)
	python scripts/start_demo.py

prepare:          ## everything except starting the app
	python scripts/start_demo.py --no-start

test-unit:
	cd backend && $(UV) pytest tests/unit -q

test-integration: ## needs `docker compose up -d`
	cd backend && $(UV) pytest tests/integration -q

test: test-unit test-integration
	cd frontend && npm run lint && npx vitest run

lint:
	$(UV) ruff check backend/app backend/tests scripts

frontend:
	cd frontend && npm ci && npm run build

bench:            ## API must be stopped; simulator running
	$(UV) python scripts/benchmark/run_benchmark.py --all

faults:           ## API + simulator running
	$(UV) python scripts/benchmark/fault_injection.py

soak:             ## API running
	$(UV) python scripts/benchmark/soak.py --hours 1

down:
	docker compose down
