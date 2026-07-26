# Synapse

Synapse is a working System 1 episodic-memory POC. It replays mocked Slack and meeting
events, interprets each event with a local model, persists the original episode and its
provisional observation, and indexes the episode in Mem0 for semantic retrieval.

## Current Flow

```mermaid
flowchart LR
	A[Mock Slack and meetings] --> B[System 1 event processing]
	B --> C[Gemma 4 on Cactus]
	C --> D[PostgreSQL episodes and observations]
	D --> E[Mem0 and pgvector index]
	D --> F[Awake workbench]
```

System 1 currently extracts:

- A compact episode summary
- A short intent label
- Salience from 0 to 1
- Mentioned entities
- Retrieval tags
- Provider confidence and latency

These observations are provisional. The current implementation does not consolidate them
into a semantic knowledge graph.

## Implemented Components

### Backend

- FastAPI application with health, configuration, episode-listing, and mock-replay routes
- SQLAlchemy models for source/retrieval episodes, intent observations, and recall events
- Alembic migration that creates the PostgreSQL schema and enables pgvector
- Duplicate-safe ingestion using stable source identifiers
- Pending-memory reconciliation when an episode exists but has not been indexed
- Worker container with database connectivity for background-process scaffolding

### Local Inference

- Cactus 2.0.1 running natively on macOS with Apple Metal
- `google/gemma-4-E2B-it` for System 1 structured interpretation
- The same Gemma server for 1,536-dimensional embeddings
- OpenAI-compatible chat, model, and embedding endpoints on port `8080`
- Deterministic interpretation fallback when Cactus is unavailable

### Memory and Storage

- PostgreSQL 18 with pgvector 0.8.5 in Docker
- Mem0 OSS embedded in the FastAPI process
- Mem0 collection `synapse_episodes_gemma_1536`
- Exact Mem0 writes with `infer=False`; Synapse controls the stored summaries
- Mem0 hybrid semantic, keyword, and entity-aware retrieval support
- In-memory Mem0 operation history; durable application state remains in PostgreSQL
- Persistent PostgreSQL Docker volume

### Workbench

- Responsive React 19 and TypeScript interface served by nginx
- Awake/System 1 episode timeline
- PostgreSQL and Cactus status indicators
- Mock replay and refresh controls
- Mem0 indexed/pending state per episode
- Placeholder navigation for Sleep and Procedural views; those views have no behavior yet

## Mock Dataset

The deterministic fixture contains five episodes covering:

- The decision to name the authentication database Titan
- Titan and Atlas API migration context across Slack and a meeting
- A project-status request
- A session-backfill blocker
- A proposal to use Crustdata during memory consolidation

Replaying the fixture multiple times does not duplicate episodes. With Cactus running, all
five episodes are indexed in Mem0/pgvector.

## Run Locally

Prerequisites:

- macOS on Apple Silicon
- Docker Desktop
- Cactus installed with `brew install cactus-compute/cactus/cactus`

Create the local environment file:

```sh
cp .env.example .env
```

Start native Gemma in a separate terminal and leave it running:

```sh
make cactus
```

Start the Docker stack and replay the fixture:

```sh
make up
make replay
```

Open:

- Workbench: http://localhost:3000
- FastAPI: http://localhost:8000
- OpenAPI: http://localhost:8000/docs

The Docker containers access native Cactus through `host.docker.internal`. If Cactus is not
running, the application remains available in degraded mode and records episodes with
pending Mem0 indexing. Running the replay again after Cactus starts reconciles them.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | PostgreSQL and Cactus availability plus active models |
| `GET` | `/api/config` | Public model, embedding dimension, and retrieval threshold |
| `GET` | `/api/episodes` | Latest persisted episodes and Mem0 indexing state |
| `POST` | `/api/demo/replay` | Idempotently process and reconcile the mock event stream |

## Docker Services

| Service | Current responsibility |
| --- | --- |
| `db` | PostgreSQL 18 and pgvector |
| `migrate` | One-shot Alembic schema upgrade |
| `api` | FastAPI, System 1, Cactus client, and Mem0 |
| `worker` | Background worker process and database heartbeat |
| `web` | nginx-hosted React build and `/api` proxy |
| `test` | Container profile for pytest |

## Commands

```sh
make up       # Build and start the Docker stack
make cactus   # Start native Gemma on Cactus/Metal
make replay   # Process or reconcile the mock event stream
make logs     # Follow API, worker, and web logs
make lint     # Run Python and frontend static checks
make test     # Run backend tests and frontend validation
make down     # Stop containers while preserving PostgreSQL data
make reset    # Stop containers and delete the PostgreSQL volume
```

## Validation Status

The current implementation passes:

- Ruff
- Strict mypy
- Pytest
- ESLint
- TypeScript compilation
- Vite production build
- Alembic migration against PostgreSQL 18
- Docker Compose health checks
- Duplicate replay and pending-memory reconciliation
- Native Gemma chat and embedding smoke tests

## Current Boundaries

The repository does not yet implement semantic graph consolidation, Crustdata API calls,
episodic-to-knowledge-base fallback retrieval, pattern detection, procedural skill
synthesis, or functional MCP tools. The Sleep and Procedural UI entries are placeholders.