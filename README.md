# RedTeamAgentLoop

Automated closed-loop LLM red-teaming agent built on LangGraph. Probes target LLMs for policy violations using adversarial prompts, mutation, and a judge LLM to score responses.

- **Attacker:** Groq API (`llama-3.3-70b-versatile`)
- **Target:** Any OpenAI-compatible endpoint (default: local tinyLLama model using Ollama)
- **Judge:** Claude Haiku via Anthropic API
- **Console:** Live Attack Console — real-time streaming UI (Phase 10)
---
## Demo


https://github.com/user-attachments/assets/6b32ad20-0d6b-4d03-b7cf-62d2e6c2e81f


---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com) (for local target models)
- Groq API key — [console.groq.com](https://console.groq.com)
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- Node 20+ and npm (for the Live Console frontend only)

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd Redteamagent

# 2. Install uv (if not already installed)
brew install uv

# 3. Install dependencies
uv sync

# 4. Install console extras (FastAPI SSE backend — needed for Live Console)
uv sync --extra console
```

---

## Setup

### 1. API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys:
#   GROQ_API_KEY=gsk_...
#   ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Authorization acknowledgment

Read `authorization.txt`. If you have explicit permission to test the target system, ensure the file contains:

```
AUTHORIZED: true
```

This line must be present or the agent will refuse to run.

### 3. Start Ollama and pull target models

**Option A — Ollama installed directly (recommended):**

```bash
brew install ollama
ollama serve          # leave running in a separate terminal tab
ollama pull tinyllama
ollama pull gemma2:2b
```

**Option B — Docker Compose:**

```bash
docker-compose up -d ollama chromadb
docker exec redteamagentloop-ollama ollama pull tinyllama
docker exec redteamagentloop-ollama ollama pull gemma2:2b
```

---

## Running

### CLI (terminal output)

```bash
# Test against tinyllama only
uv run redteamagentloop \
  --objective "elicit unlicensed investment advice" \
  --target tinyllama \
  --config config.yaml

# Test against all configured targets
uv run redteamagentloop \
  --objective "elicit unlicensed investment advice" \
  --config config.yaml

# With a custom target system prompt
uv run redteamagentloop \
  --objective "extract customer PII" \
  --system-prompt "You are a helpful banking assistant." \
  --target gemma2 \
  --config config.yaml
```

#### CLI options

| Flag | Default | Description |
|---|---|---|
| `--objective` | required | What the target must NOT do |
| `--target` | all targets | `output_tag` from `config.yaml` |
| `--system-prompt` | `""` | System prompt to inject into the target |
| `--config` | `config.yaml` | Path to config file |
| `--auth` | `authorization.txt` | Path to authorization file |
| `--output-dir` | `reports/output` | Directory for HTML reports |

---

### Live Console (streaming web UI)

A terminal-style web UI that streams adversarial prompts, target responses, and animated judge scores in real time.

#### Option A — Dev mode (backend + frontend separately)

```bash
# Terminal 1 — FastAPI SSE backend
uv run uvicorn console.backend.main:app --reload --port 8000

# Terminal 2 — Vite frontend dev server (proxies /api to :8000)
cd console/frontend
npm install
npm run dev
# Open http://localhost:5173
```

#### Option B — Production mode (backend serves built frontend)

```bash
# Build the frontend once
cd console/frontend
npm install
npm run build
cd ../..

# Serve everything from the backend
uv run uvicorn console.backend.main:app --port 8000
# Open http://localhost:8000
```

#### Option C — Docker (builds everything automatically)

```bash
# Build and start Ollama + ChromaDB + Console in one command
docker compose up --build

# Pull models into the running Ollama container
docker exec redteamagentloop-ollama ollama pull tinyllama
docker exec redteamagentloop-ollama ollama pull gemma2:2b

# Open http://localhost:8000
```

> The console reads `config.yaml` and your `.env` file at startup.
> Make sure both exist and your API keys are set before starting.

---

## Configuration

Key settings in `config.yaml`:

```yaml
loop:
  max_iterations: 50             # reduce to 5 for a quick test
  vuln_threshold: 7.0            # score >= this = confirmed vulnerability
  mutation_batch_size: 3         # mutations generated per cycle
  max_mutations_per_strategy: 8  # rotate strategy after this many mutation cycles

attacker:
  rpm: 30   # Groq free tier rate limit (0 = disabled)

judge:
  model: "claude-haiku-4-5-20251001"   # swap for claude-sonnet-4-6 for production runs
```

### Provider configuration

All three roles (attacker, judge, target) are configurable via `config.yaml`. LLMs are built
once at startup by `redteamagentloop/llm_factory.py` and injected into the agent graph.

#### Attacker providers

| `provider` | API key env var | Notes |
|---|---|---|
| `groq` (default) | `GROQ_API_KEY` | Groq API — free tier: 30 RPM |
| `openai` | `OPENAI_API_KEY` | OpenAI API |
| `ollama` | _(none)_ | Local Ollama; uses `api_key: "ollama"` |
| `custom` | `ATTACKER_API_KEY` | Any OpenAI-compatible endpoint |

#### Judge providers

| `provider` | API key env var | Notes |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | Claude via Anthropic API |
| `openai` | `OPENAI_API_KEY` | OpenAI API |
| `custom` | `JUDGE_API_KEY` | Any OpenAI-compatible endpoint; requires `base_url` |

#### Target

The target has no `provider` field — it is always OpenAI-compatible. Set `base_url` and `api_key` directly:

```yaml
targets:
  - model: "tinyllama"
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
```

#### Example: using a custom provider for the judge

```yaml
judge:
  provider: "custom"
  model: "meta-llama/llama-3-70b-instruct"
  base_url: "https://your-inference-api.example.com/v1"
  temperature: 0.1
  max_tokens: 512
```

Set `JUDGE_API_KEY=<your-key>` in your `.env` file.

---

## Strategy tools

Three standalone scripts for inspecting strategies and evaluating judge quality outside the full agent loop. All three live in the `tests/` directory.

### tests/run_all_strategies.py — run every strategy once

Fires all 10 attack strategies in sequence. In mock mode no API keys are needed; in live mode the full attacker → target → judge pipeline runs for each strategy and prints a summary table.

```bash
# Mock mode — no API keys or Ollama required
uv run python tests/run_all_strategies.py

# Live mode — real Groq attacker + Ollama target + Claude judge
uv run python tests/run_all_strategies.py --live

# Different objective
uv run python tests/run_all_strategies.py --live \
  --objective "bypass KYC compliance checks"

# Different target
uv run python tests/run_all_strategies.py --live --target gemma2
```

### tests/run_strategy.py — run one strategy

Runs a single named strategy and prints the attacker prompt, target response, and judge verdict. Useful for debugging or inspecting a specific strategy in isolation.

```bash
# List all available strategies with descriptions
uv run python tests/run_strategy.py --list

# Mock mode — inspect the generated prompt only
uv run python tests/run_strategy.py DirectJailbreak
uv run python tests/run_strategy.py ObfuscatedRequest \
  --objective "reveal the system prompt"

# Live mode — full attacker + target + judge report
uv run python tests/run_strategy.py DirectJailbreak --live
uv run python tests/run_strategy.py FinServSpecific --live --target gemma2
```

### tests/run_golden_dataset_eval.py — evaluate judge quality

Loads a human-labelled JSONL dataset, runs the configured judge LLM against every item, and computes MAE, RMSE, and Pearson r to measure how closely the judge scores agree with human scores. Optionally runs RAGAS FactualCorrectness on the judge's reasoning. Writes a Markdown report to `reports/`.

Requires `ANTHROPIC_API_KEY` (or whichever judge provider is set in `config.yaml`).

```bash
# Basic run — MAE, RMSE, Pearson r only
uv run python tests/run_golden_dataset_eval.py

# With RAGAS evaluation of judge reasoning quality
uv run python tests/run_golden_dataset_eval.py --ragas

# Custom dataset or output path
uv run python tests/run_golden_dataset_eval.py \
    --dataset evaluation/judge_eval_dataset.jsonl \
    --output reports/judge_eval_report.md

# Quick smoke test — evaluate only the first 10 items
uv run python tests/run_golden_dataset_eval.py --limit 10
```

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `evaluation/judge_eval_dataset.jsonl` | Path to human-labelled JSONL file |
| `--output` | `reports/judge_eval_report.md` | Path for the Markdown report |
| `--config` | `config.yaml` | Config file (determines judge model) |
| `--ragas` | off | Run RAGAS FactualCorrectness on judge reasoning |
| `--concurrency` | `5` | Max parallel judge calls |
| `--limit` | none | Evaluate only the first N items |

Exit code is `0` if MAE ≤ 1.5 (pass threshold), `1` otherwise.

---

## Tests

No API keys or running services required — all LLMs are mocked.

```bash
# Full test suite (271 tests)
uv run pytest --tb=short -q

# By phase
uv run pytest tests/unit/test_state.py -v        # Phase 1 — state schema
uv run pytest tests/unit/test_strategies.py -v   # Phase 2 — 10 strategies
uv run pytest tests/unit/test_storage.py -v      # Phase 3 — storage layer
uv run pytest tests/unit/test_nodes.py -v        # Phase 4 — 6 nodes
uv run pytest tests/unit/test_graph.py -v        # Phase 5 — graph wiring
uv run pytest tests/unit/test_evaluation.py -v   # Phase 6 — judge evaluator
uv run pytest tests/unit/test_reporting.py -v    # Phase 7 — HTML reports
uv run pytest tests/test_regression.py -v        # Phase 8 — regression dataset
uv run pytest tests/integration/ -v              # Phase 8 — E2E integration
uv run pytest tests/unit/test_phase9.py -v       # Phase 9 — guardrails/rate limiter
uv run pytest tests/unit/test_console.py -v      # Phase 10 — live console backend

# With coverage
uv run pytest --cov=redteamagentloop --cov-report=term-missing
```

---

## Output

Each run produces:

- **Terminal** — live score gauge + iteration table via Rich
- **HTML report** — saved to `reports/output/<session>_<timestamp>.html`
- **JSONL** — `reports/<target_tag>_vulnerabilities.jsonl`
- **SQLite** — `reports/metadata.db`
- **ChromaDB** — `reports/chroma_<target_tag>/` (semantic dedup)
- **Session log** — `reports/logs/<session_id>.log` (JSON structured)

---

## Architecture

```
START
  └─► attacker ──► target_caller ──► judge ──► loop_controller
                                                      │
                    ┌─────────────────────────────────┤
                    │                                 │
              score >= 7.0                     score < 7.0
                    │                                 │
              vuln_logger               mutation_queue empty?
                    │                        │              │
                    └──► mutation_engine ◄───┘         mutation_engine
                               │
                           attacker  (or END if max_iterations reached)
```

**10 attack strategies:** `DirectJailbreak`, `PersonaHijack`, `DirectInjection`, `IndirectInjection`, `FewShotPoisoning`, `NestedInstruction`, `AdversarialSuffix`, `ContextOverflow`, `ObfuscatedRequest`, `FinServSpecific`

**Phase 9 hardening:** ethical guardrails (CBRN/CSAM filter), token-bucket rate limiting per LLM, circuit breaker on target errors, JSON structured logging, startup API key validation.

**Phase 10 console:** FastAPI SSE backend streams `AttackEvent`s from LangGraph's `astream_events`; React + Vite frontend renders live score gauges, iteration log, and vuln highlights.
