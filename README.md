# pi-kg

Build a clean, canonical, well-connected **knowledge graph** from a document corpus using a
minimal [Pi](https://github.com/earendil-works/pi-coding-agent) harness driven **entirely by a
system prompt** — no bespoke extraction pipeline, no hard-coded dedup. The agent (a local
`Qwen3.6-35B-A3B-MTP` on a single L4) reads the corpus and uses embedding tools to canonicalize
and deduplicate entities/predicates itself, as test-time compute.

Same philosophy as [searchbox](https://github.com/hanxiao/searchbox): Pi is already a complete
agent (it loops, calls tools, auto-compacts). We add exactly one thing over vanilla Pi — keep
nudging (`Continue.`) until the turn budget is spent. Everything else lives in the **system
prompt**.

## Why

The KG is used downstream to find long multi-hop fact chains (for building hard, verifiable,
private evals for agentic search). That only works if the SAME real-world entity is ONE node
across every document. The hard part is therefore not extraction but **canonicalization /
deduplication** of entity nodes and predicates. This repo tests whether a local agent, given only
a system prompt and embedding tools, can produce a graph clean enough for multi-hop path mining.

## Architecture

```
run_pi_kg.py                 thin orchestrator: boot sidecar, launch pi --mode rpc,
                             send the task once, nudge Continue. until turn budget spent
prompts/
  kg_system_steps.txt        VARIANT A: explicit step-by-step workflow (STEP 1..7)
  kg_system_outcome.txt      VARIANT B: only defines the required OUTCOME + quality bars;
                             the agent figures out its own steps
pi/extensions/kg-tools.ts    registers the embedding tools as pi tools (from tools-catalog.json)
pi/tools-catalog.json        tool specs (embed_texts, similarity, cluster, select_diverse, rerank)
server/dataroom_service.py   local embedding/rerank sidecar (jina-embeddings-v5 + reranker-v3);
                             endpoints /embed /similarity /cluster /deduplicate /rerank
```

The model's tools: Pi built-ins (`bash`, `grep`, `read`, `write`, `edit`, `ls`) + the embedding
primitives above. Deliverable: `KG.jsonl` (one canonical edge per line) + `KG_STATS.md`.

## The A/B test (PROMPT_VARIANT)

The central experiment: **do you tell the agent the STEPS, or only the desired RESULT?**

- `PROMPT_VARIANT=steps`   — spell out the procedure (inventory -> extract per-doc -> cluster-embed
  -> canonicalize entities -> canonicalize predicates -> merge edges -> verify connectivity ->
  finalize). Deterministic, but constrains the agent to our recipe.
- `PROMPT_VARIANT=outcome` — define only what "done" looks like (valid JSONL, grounded facts,
  canonical & deduplicated nodes, no duplicate edges, well-connected graph) and let the agent
  figure out how. Tests whether a strong agent self-organizes a better pipeline.

Run both on the same corpus and compare KG quality (see Metrics). Thinking is **ON** (`high`) in
both — the canonicalization judgment is exactly the kind of reasoning thinking-mode helps.

## Model backend

`Qwen3.6-35B-A3B-MTP` served per
[hanxiao/Qwen3.6-35B-A3B-MTP-L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) on a single
NVIDIA L4 (Q4_K_XL + MTP, ~92-100 tok/s). `CONTEXT_WINDOW` **must** match the server's
`--ctx-size` (56320 on the L4) or Pi compacts too late and the server rejects over-long requests.

## Setup

```bash
# 1. Pi (latest)
npm install -g @earendil-works/pi-coding-agent@latest   # pinned tested version in PI_VERSION

# 2. Python deps for the sidecar (embedding/rerank)
python3 -m venv .venv && . .venv/bin/activate
pip install -r server/requirements.txt

# 3. Point at your model server + configure (see .env.example)
cp .env.example .env    # edit LLAMA_URL etc, then: set -a; . ./.env; set +a
```

## Run

```bash
# Variant A (steps)
PROMPT_VARIANT=steps   python run_pi_kg.py --dataroom path/to/corpus.zip --budget 12 --out out/steps
# Variant B (outcome)
PROMPT_VARIANT=outcome python run_pi_kg.py --dataroom path/to/corpus.zip --budget 12 --out out/outcome
```

`--dataroom` accepts a `.zip` or a folder. Output: `out/<name>/KG.jsonl` + `KG_STATS.md`.

## Metrics (how to judge A vs B)

From `KG_STATS.md` / `KG.jsonl`:
- total edges, total canonical nodes
- **cross-document entities** (entities appearing in >=2 docs) — higher = better connectivity
- **largest connected component** fraction — higher = graph mines longer multi-hop paths
- duplicate-node rate (surface variants left unmerged) — lower = better canonicalization
- duplicate-edge rate — should be ~0 after merge
- groundedness (fraction of facts whose evidence_span is a verbatim substring of its source doc)

## License

MIT
