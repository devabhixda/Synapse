.PHONY: up down build logs replay test lint reset cactus

up:
	docker compose up -d --build --wait

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f api worker web

replay:
	curl --fail --request POST http://localhost:$${SYNAPSE_WEB_PORT:-3000}/api/demo/replay

test:
	docker compose run --rm --build test
	npm --prefix apps/web run lint
	npm --prefix apps/web run build

lint:
	docker compose run --rm api ruff check src tests migrations
	docker compose run --rm api mypy src
	npm --prefix apps/web run lint

reset:
	docker compose down --volumes

cactus:
	sh scripts/cactus-host.sh