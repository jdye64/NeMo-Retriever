# NeMo Retriever Development Compose Helpers

These files start development-only helper services. Hosted inference remains
the zero-profile default. Kubernetes Helm charts and the HTTP Retriever
service Compose stack are no longer supported.

Run commands from the repository root.

## Local judge

`judge.compose.yaml` starts a local OpenAI-compatible Nemotron NIM for
`retriever skill-eval` runs that use a local judge endpoint. Set `NGC_API_KEY`
or `NIM_NGC_API_KEY` before starting it, then authenticate to NGC and launch
the helper:

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker compose -f nemo_retriever/dev/compose/judge.compose.yaml up -d judge
```

Point `judge.api_base` at `http://localhost:8000/v1` in the skill-eval
configuration. Override the host port with `JUDGE_HTTP_PORT` when needed.

## Neo4j

`neo4j.compose.yaml` starts the graph development database. Set
`NEO4J_PASSWORD` in the environment or a repository-root `.env` file before
starting it; `NEO4J_USERNAME` defaults to `neo4j`.

```bash
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml up -d neo4j
```
