# ---------------------------------------------------------------------------
# Convenience targets. Usage: `make demo`, `make test`, `make load`, `make down`.
# (Requires Docker + Docker Compose. On Windows, use Git Bash or run_demo.ps1.)
# ---------------------------------------------------------------------------
SCALE ?= 1

.PHONY: demo up down logs test load build scale clean

## demo: build + start the full stack (API + nginx + UI), then wait for health
demo:
	./run_demo.sh $(SCALE)

## build: build the Docker image only
build:
	docker compose build

## up: start the stack in the background
up:
	docker compose up -d --scale api=$(SCALE)

## scale: (re)start with N api replicas, e.g. `make scale SCALE=4`
scale:
	docker compose up -d --scale api=$(SCALE)

## down: stop and remove containers
down:
	docker compose down

## logs: follow logs
logs:
	docker compose logs -f

## test: run the manual endpoint walkthrough against the running API
test:
	./test_api.sh

## load: start Locust against the load balancer (open http://localhost:8089)
load:
	locust -f locustfile.py --host http://localhost:8000

## clean: stop stack and remove uploaded batches / job state
clean: down
	rm -rf uploads/* jobs/* && touch uploads/.gitkeep jobs/.gitkeep
