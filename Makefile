.PHONY: up down build logs replay slack-preview slack-import test lint reset

up:
	docker compose up -d --build --wait

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f api worker web

replay:
	@test -n "$${SYNAPSE_GROUP_ID}" || (echo "Set SYNAPSE_GROUP_ID to the account to replay into" && exit 1)
	curl --fail --request POST "http://localhost:$${SYNAPSE_WEB_PORT:-3000}/api/demo/replay?group_id=$${SYNAPSE_GROUP_ID}"

slack-preview:
	@test -n "$${SLACK_EXPORT_DIR}" || (echo "Set SLACK_EXPORT_DIR to the extracted Slack export directory" && exit 1)
	docker compose run --rm -T -v "$${SLACK_EXPORT_DIR}:/data/slack-export:ro" api python -m synapse.sources.slack_import --export-root /data/slack-export --channel all-docwise --channel scribe-iteration --max-events 40 --max-per-channel 30 --min-score 4 --dry-run

slack-import:
	@test -n "$${SLACK_EXPORT_DIR}" || (echo "Set SLACK_EXPORT_DIR to the extracted Slack export directory" && exit 1)
	@test -n "$${SYNAPSE_GROUP_ID}" || (echo "Set SYNAPSE_GROUP_ID to the account this export belongs to" && exit 1)
	docker compose run --rm -T -v "$${SLACK_EXPORT_DIR}:/data/slack-export:ro" api python -m synapse.sources.slack_import --export-root /data/slack-export --group-id "$${SYNAPSE_GROUP_ID}" --channel all-docwise --channel scribe-iteration --max-events 40 --max-per-channel 30 --min-score 4

# Tests build a SQLite file per test in a temp directory, so this no longer
# has to stand up a database first. Run `docker compose run --rm --build test`
# to run the same suite inside the pinned image.
test:
	pytest
	npm --prefix apps/web run lint
	npm --prefix apps/web run build

lint:
	docker compose run --rm api ruff check src tests migrations
	docker compose run --rm api mypy src
	npm --prefix apps/web run lint

reset:
	docker compose down --volumes