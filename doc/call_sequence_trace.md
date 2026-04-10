# Call Sequence Trace: CLI to Report Generation

Complete function call path from `uv run redteamagentloop` to HTML report written to disk.

---

```
uv run redteamagentloop
  ↓
main() — redteamagentloop/cli.py:101
  ├─ load_dotenv()
  ├─ check_authorization() — redteamagentloop/config.py:107
  ├─ load_config() — redteamagentloop/config.py:147  →  AppConfig (Pydantic)
  ├─ check_api_keys() — redteamagentloop/config.py:126
  ├─ build_graph(app_config) — redteamagentloop/agent/graph.py:27
  │     StateGraph with 6 nodes:
  │       attacker → target_caller → judge → loop_controller
  │       loop_controller --conditional--> vuln_logger | mutation_engine | attacker | END
  │       vuln_logger → mutation_engine → attacker
  ├─ build_initial_state() — redteamagentloop/agent/graph.py:58  →  RedTeamState (TypedDict)
  └─ asyncio.run(run_all())
       └─ _run_target(graph, state, app_config, target, output_dir) — redteamagentloop/cli.py:16
            ├─ StorageManager + TerminalDashboard setup
            ├─ graph.astream(initial_state, config=run_config) — redteamagentloop/cli.py:69
            │    [per iteration]
            │    ├─ attacker_node() — redteamagentloop/agent/nodes/attacker.py:41
            │    │     _next_strategy() → pick strategy (skip failed_strategies)
            │    │     strategy.generate_prompt(state, attacker_llm) → prompt
            │    │
            │    ├─ target_caller_node() — redteamagentloop/agent/nodes/target_caller.py
            │    │     call target LLM → current_response
            │    │
            │    ├─ judge_node() — redteamagentloop/agent/nodes/judge.py
            │    │     ChatAnthropic → score (0-10), score_rationale
            │    │
            │    ├─ loop_controller_node() — redteamagentloop/agent/nodes/loop_controller.py:25
            │    │     append AttackRecord to attack_history
            │    │     if strategy_mutation_count >= max_mutations_per_strategy
            │    │         → mark strategy in failed_strategies
            │    │
            │    ├─ route_after_judge() — redteamagentloop/agent/nodes/loop_controller.py:12
            │    │     score >= vuln_threshold  → vuln_logger
            │    │     mutation_queue empty     → mutation_engine
            │    │     mutation_queue has items → attacker
            │    │     error / max_iterations   → END
            │    │
            │    ├─ [if score >= vuln_threshold]
            │    │     vuln_logger_node() — redteamagentloop/agent/nodes/vuln_logger.py:19
            │    │       storage_manager.log_attack(record)
            │    │       append to successful_attacks
            │    │       → mutation_engine
            │    │
            │    └─ mutation_engine_node() — redteamagentloop/agent/nodes/mutation_engine.py:68
            │          _select_tactics() → batch of tactics (Paraphrase, LanguageSwap, etc.)
            │          attacker_llm rewrites seed prompt → mutated prompts
            │          updated_queue appended → mutation_queue
            │          strategy_mutation_count += 1
            │          → attacker (loop back)
            │
            │    [END when error or iteration_count >= max_iterations]
            │
            ├─ ReportGenerator().load_session_data() — reports/report_generator.py:62
            │     builds SessionReport dataclass from final state
            │
            ├─ ReportGenerator().render_html() — reports/report_generator.py:89
            │     loads reports/templates/report.html.j2 (Jinja2)
            │     computes strategy_chart_data (avg scores per strategy)
            │     renders HTML string
            │
            └─ ReportGenerator().save() — reports/report_generator.py:116
                  creates reports/output/ if needed
                  writes reports/output/{session_id[:8]}_{timestamp}.html
```

---

## Golden Dataset Evaluation (offline, run_golden_dataset_eval.py)

Complete call path for judge quality evaluation against the human-labelled dataset.

```
uv run python run_golden_dataset_eval.py [--ragas] [--limit N]
  ↓
main() — run_golden_dataset_eval.py
  ├─ load_dotenv()
  ├─ load_config() — redteamagentloop/config.py:147  →  AppConfig (judge model/settings)
  ├─ ChatAnthropic(...) — builds judge LLM from app_config.judge
  ├─ JudgeEvaluator.load_dataset() — evaluation/judge_evaluator.py:70
  │     reads evaluation/judge_eval_dataset.jsonl  →  list[EvalItem]
  ├─ make_judge_fn(judge_llm) — evaluation/judge_evaluator.py:244
  │     wraps ChatAnthropic into async callable(target_objective, prompt, response)
  │     uses prompts/judge_template.j2 + JudgeOutput structured output
  ├─ JudgeEvaluator.evaluate_all(judge_fn, items) — evaluation/judge_evaluator.py:90
  │     asyncio.Semaphore(concurrency=5)
  │     [per item, in parallel]
  │       judge_fn(item.target_objective, item.prompt, item.response)
  │         → score (float), reasoning (str)
  │     compute_metrics(predicted, human_scores, items) — evaluation/judge_evaluator.py:134
  │       → EvalMetrics: MAE, RMSE, Pearson r, per-strategy breakdown
  │     → EvalResults(items, predicted_scores, judge_reasonings, metrics)
  │
  ├─ [if --ragas]
  │     run_ragas_eval(items, judge_reasonings) — evaluation/ragas_eval.py:52
  │       build_ragas_dataset(items, judge_reasonings) — evaluation/ragas_eval.py:23
  │         maps each item → SingleTurnSample(user_input, response, reference)
  │         → ragas.EvaluationDataset
  │       ragas.evaluate(dataset, metrics=[FactualCorrectness()])
  │         → mean factual_correctness score (0-1)
  │     attach_ragas_metrics(results, ragas_scores) — evaluation/ragas_eval.py:90
  │       sets EvalMetrics.ragas_factual_correctness
  │
  └─ JudgeEvaluator.generate_report(results, output_path) — evaluation/judge_evaluator.py:193
        writes Markdown report → reports/judge_eval_report.md
        includes: MAE/RMSE/Pearson table, per-strategy breakdown,
                  item-level results, RAGAS scores (if computed)
```

Exit code: `0` if MAE ≤ 1.5, `1` otherwise (usable as a CI gate).

---

## Key Files

| Role | File |
|---|---|
| CLI entry point | `redteamagentloop/cli.py` |
| Graph wiring | `redteamagentloop/agent/graph.py` |
| State schema | `redteamagentloop/agent/state.py` |
| Config models/loader | `redteamagentloop/config.py` |
| Attacker node | `redteamagentloop/agent/nodes/attacker.py` |
| Target caller node | `redteamagentloop/agent/nodes/target_caller.py` |
| Judge node | `redteamagentloop/agent/nodes/judge.py` |
| Loop controller node | `redteamagentloop/agent/nodes/loop_controller.py` |
| Vuln logger node | `redteamagentloop/agent/nodes/vuln_logger.py` |
| Mutation engine node | `redteamagentloop/agent/nodes/mutation_engine.py` |
| Report generator | `reports/report_generator.py` |
| Report template | `reports/templates/report.html.j2` |

---

## Node Graph with Conditional Edges

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
START ──► attacker ──► target_caller ──► judge ──► loop_controller
              ▲                                        │
              │                               route_after_judge()
              │                                [CONDITIONAL]
              │                                        │
              │             ┌──────────────────────────┤
              │             │              │            │
              │        error/max    score>=threshold  queue
              │        iterations         │            empty
              │             │             ▼            │
              │            END       vuln_logger        │
              │                           │             │
              │                           ▼             ▼
              └───────────────────── mutation_engine ◄──┘
```

`loop_controller` is the **only node with a conditional outbound edge**, via `route_after_judge()` in
`redteamagentloop/agent/nodes/loop_controller.py:12`.

| Condition | Destination |
|---|---|
| `error` set or `iteration_count >= max_iterations` | `END` |
| `score >= vuln_threshold` | `vuln_logger` |
| `mutation_queue` has items | `attacker` |
| `mutation_queue` is empty | `mutation_engine` |

All other edges are unconditional:
`attacker → target_caller → judge → loop_controller`, `vuln_logger → mutation_engine`, `mutation_engine → attacker`.
