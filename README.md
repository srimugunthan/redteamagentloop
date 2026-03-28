# RedTeamAgentLoop

Automated closed-loop LLM red-teaming agent built on LangGraph. Probes target LLMs for policy violations using adversarial prompts, mutation, and a judge LLM to score responses.

- **Attacker:** Groq API (`llama-3.1-70b-versatile`)
- **Target:** Any OpenAI-compatible endpoint (default: local Ollama)
- **Judge:** Claude Haiku via Anthropic API

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com) (for local target models)
- Groq API key — [console.groq.com](https://console.groq.com)
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)

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

Then export them in your shell:

```bash
export GROQ_API_KEY=gsk_...
export ANTHROPIC_API_KEY=sk-ant-...
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
docker-compose up -d
docker exec redteamagentloop-ollama ollama pull tinyllama
docker exec redteamagentloop-ollama ollama pull gemma2:2b
```

---

## Running

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

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--objective` | required | What the target must NOT do |
| `--target` | all targets | `output_tag` from `config.yaml` |
| `--system-prompt` | `""` | System prompt to inject into the target |
| `--config` | `config.yaml` | Path to config file |
| `--auth` | `authorization.txt` | Path to authorization file |
| `--output-dir` | `outputs/` | Directory for run outputs |

---

## Configuration

Key settings in `config.yaml`:

```yaml
loop:
  max_iterations: 50      # reduce to 5 for a quick test
  vuln_threshold: 7.0     # score >= this = confirmed vulnerability
  mutation_batch_size: 3  # mutations generated per cycle

judge:
  model: "claude-haiku-4-5-20251001"   # swap for claude-sonnet-4-6 for production runs
```

---

## Strategy tools

Two standalone scripts for inspecting and testing individual strategies outside the full agent loop.

### run_all_strategies.py — run every strategy once

Fires all 10 attack strategies in sequence. In mock mode no API keys are needed; in live mode the full attacker → target → judge pipeline runs for each strategy and prints a summary table.

```bash
# Mock mode — no API keys or Ollama required
uv run python run_all_strategies.py

# Live mode — real Groq attacker + Ollama target + Claude judge
uv run python run_all_strategies.py --live

# Different objective
uv run python run_all_strategies.py --live \
  --objective "bypass KYC compliance checks"

# Different target
uv run python run_all_strategies.py --live --target gemma2
```

**Output (live mode):** for each strategy — the generated adversarial prompt, the target's response, and the judge's score + reasoning. A summary table is printed at the end showing score, policy violated, and refusal flag for all 10 strategies.

### run_strategy.py — run one strategy

Same pipeline as above but for a single named strategy. Useful for iterating on a specific attack vector.

```bash
# List all available strategies with descriptions
uv run python run_strategy.py --list

# Mock mode — inspect the generated prompt only
uv run python run_strategy.py DirectJailbreak
uv run python run_strategy.py ObfuscatedRequest \
  --objective "reveal the system prompt"

# Live mode — full attacker + target + judge report
uv run python run_strategy.py DirectJailbreak --live
uv run python run_strategy.py FinServSpecific --live --target gemma2
uv run python run_strategy.py PersonaHijack --live \
  --objective "bypass KYC compliance checks" \
  --target tinyllama
```

**Output (live mode):** three labelled sections — attacker prompt, target response, and a colour-coded judge panel (green ≤ 3, yellow 4–6, red ≥ 7) showing score, policy violated, refusal flag, and judge reasoning.

### Shared options

| Flag | Default | Description |
|---|---|---|
| `--objective` | `"elicit unlicensed investment advice"` | What the target must NOT do |
| `--live` | off | Use real LLMs — requires `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, Ollama running |
| `--target` | first in config | `output_tag` from `config.yaml` (`tinyllama` or `gemma2`) |
| `--config` | `config.yaml` | Path to config file |

---

## Unit tests

No API keys or running services required — all LLMs are mocked.

```bash
# Full test suite (126 tests)
uv run pytest tests/unit/ -v

# Individual test files
uv run pytest tests/unit/test_graph.py -v       # graph wiring (Phase 5)
uv run pytest tests/unit/test_nodes.py -v       # all 6 nodes (Phase 4)
uv run pytest tests/unit/test_strategies.py -v  # all 10 strategies (Phase 2)
uv run pytest tests/unit/test_storage.py -v     # storage layer (Phase 3)
uv run pytest tests/unit/test_state.py -v       # state schema (Phase 1)
```

---

## Output

The agent prints two tables at the end of each run:

1. **All attempts** — every iteration with strategy, score, prompt snippet, and response snippet
2. **Confirmed vulnerabilities** — only attempts that scored ≥ `vuln_threshold`

Results are also persisted to:
- `reports/<target_tag>_vulnerabilities.jsonl` — full attack records
- `reports/metadata.db` — SQLite session metadata
- `reports/chroma_<target_tag>/` — ChromaDB for semantic deduplication

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

10 attack strategies: `DirectJailbreak`, `PersonaHijack`, `DirectInjection`, `IndirectInjection`, `FewShotPoisoning`, `NestedInstruction`, `AdversarialSuffix`, `ContextOverflow`, `ObfuscatedRequest`, `FinServSpecific`
