# Attack Strategy Bank — Framework Coverage Analysis

> Mapped against OWASP LLM Top 10 (2025), NIST AI RMF, and MITRE ATLAS

---

## The Full Comparison

| Dimension | promptfoo | PyRIT | garak | DeepTeam | **RedTeamAgentLoop** |
|---|---|---|---|---|---|
| **Attack auto-gen** | Hybrid | Full (attacker LLM) | Static corpus | Full (LLM synthesizer) | **Full (Groq attacker LLM)** |
| **Mutation / adaptation** | None | Limited (converters) | Static only | Multi-turn progression | **Dedicated mutation node in graph** |
| **Closed feedback loop** | No | Partial (orchestrators) | No | Partial | **Yes — native LangGraph loop** |
| **Attack breadth** | High (plugin system) | High (composable) | Highest (150+ probes) | High (50+ vulns) | **Moderate (10 strategies)** |
| **Domain specificity** | Generic | Generic | Generic | Generic | **FinServ-specific strategy built in** |
| **Local model support** | Limited | Limited | Yes (HuggingFace) | Yes (Ollama) | **Yes (Ollama first-class)** |

---

## Coverage Map

| Strategy | Family | OWASP LLM Top 10 | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|---|
| DirectJailbreak | Role-play | ✅ LLM01 Prompt Injection | ✅ AML.T0051 | 🟡 GOVERN 1.2 |
| PersonaHijack | Role-play | ✅ LLM01 / 🟡 LLM09 Misinformation | ✅ AML.T0051.001 | 🟡 MAP 5.2 |
| IndirectInjection | Prompt Injection | ✅ LLM02 Indirect Injection | ✅ AML.T0054 | 🟡 MANAGE 2.4 |
| DirectInjection | Prompt Injection | ✅ LLM01 | ✅ AML.T0051 | 🟡 MANAGE 2.4 |
| AdversarialSuffix | Token Manip. | 🟡 LLM01 (implicit) | ✅ AML.T0043 Craft Adv. Examples | ❌ Not addressed |
| ObfuscatedRequest | Encoding | 🟡 LLM01 (variant) | 🟡 AML.T0051 (implicit) | ❌ Not addressed |
| ContextOverflow | Context Manip. | 🟡 LLM04 Data/Model Poisoning | 🟡 AML.T0051 (variant) | ❌ Not addressed |
| FewShotPoisoning | In-context Learning | 🟡 LLM04 | ✅ AML.T0047 ICL Manip. | ❌ Not addressed |
| NestedInstruction | Structural | 🟡 LLM01 (variant) | 🟡 AML.T0051 (implicit) | ❌ Not addressed |
| FinServSpecific | Domain-Specific | 🟡 LLM09 Misinformation | ❌ No direct mapping | ❌ Not addressed |

**Legend:** ✅ Covered | 🟡 Partial | ❌ Not mapped / Gap

---

## Coverage Summary

Of the 10 strategies, only **4 have strong coverage** across all three frameworks:

- `DirectJailbreak`
- `PersonaHijack`
- `IndirectInjection`
- `DirectInjection`

These are the "classic" attacks that frameworks were designed around. The remaining 6 are either partially covered or fall into genuine gaps — which is where the red team bank adds real differentiation.

---

## Where Each Framework Underdelivers

### OWASP LLM Top 10 (2025)

OWASP anchors everything to LLM01 (prompt injection) and LLM02 (indirect injection), which means it tends to flatten attack variants into a single bucket. `AdversarialSuffix` (GCG-style token optimization), `ObfuscatedRequest`, and `ContextOverflow` all get collapsed into LLM01 even though their threat model and mitigations are quite different.

OWASP also has no control mappings for **inference-time context manipulation** — it was designed when attacks were more about jailbreaking via clever wording, not logit-level exploitation.

### NIST AI RMF

NIST AI RMF is a governance and process framework, not a technical threat taxonomy. It's excellent for `GOVERN` / `MAP` / `MEASURE` / `MANAGE` lifecycle controls, but has essentially no specificity about *how* attacks work at the token or prompt level.

For **5 of the 10 strategies**, the NIST mapping is "not addressed" — meaning a red team running purely against NIST controls would have no guidance on adversarial suffixes, encoding-based bypasses, or context flooding.

### MITRE ATLAS

ATLAS is the most technically grounded of the three. AML.T0043 (craft adversarial examples) maps to `AdversarialSuffix`, and AML.T0047 covers in-context learning manipulation (`FewShotPoisoning`). However, ATLAS was designed around ML systems broadly and has **no finserv regulatory threat scenarios** — so `FinServSpecific` has no ATLAS mapping at all.

---

## Critical Gaps 

### 1. No RAG Pipeline Targeting
No framework explicitly covers retrieval-augmented generation as a distinct attack surface. Garak's latent injection probes and promptfoo's RAG-specific plugins test retrieval pipelines specifically. RedTeamAgentLoop probes any OpenAI-compatible endpoint, which means RAG-specific attack surfaces (context stuffing, malicious document injection) aren't first-class citizens yet.

### 2. Attack Breadth Gap
10 strategies vs. garak's 150+ probes and DeepTeam's 50+ vulnerability types means RedTeamAgentLoop covers a narrower threat surface. For broad-spectrum baseline scanning of an unknown model, running garak first remains the better choice. RedTeamAgentLoop's advantage is depth and adaptivity on a targeted objective, not breadth of coverage.

## — What No Framework Covers
### 3. GCG / Gradient-Based Token Attacks
OWASP LLM Top 10 does not treat adversarial suffix optimization (GCG, AutoDAN) as a first-class threat. NIST AI RMF has no technical countermeasure guidance. Only MITRE ATLAS AML.T0043 covers it, but without finserv-specific threat scenarios.

### 4. Encoding / Obfuscation Variants
Base64, ROT13, leetspeak encoding are not explicitly enumerated in any framework as distinct attack vectors. They collapse into generic "prompt injection" which understates the bypass risk.

### 5. Context Window Flooding
No framework has a dedicated control or mitigation for **attention dilution via context overflow**. OWASP LLM04 (data poisoning) is the closest but addresses training-time, not inference-time context manipulation.

### 6. In-Context Few-Shot Poisoning
MITRE ATLAS AML.T0047 captures this best, but OWASP and NIST provide no controls. The fabricated-example-as-norm-setter attack is underspecified across all frameworks.

### 7. FinServ Domain-Specific Attacks ⚠️
**KYC bypass, MNPI leakage, suitability override, and unlicensed advice** are completely absent from OWASP, NIST AI RMF, and MITRE ATLAS. No framework maps regulatory compliance violations as an LLM attack surface. This is the strongest competitive differentiation for red teamers in financial services.

### 8. Agentic / Multi-Turn Attack Chaining
No framework covers multi-step attack orchestration where early turns set up later exploitation (e.g., persona priming → indirect injection → tool misuse). All frameworks treat attacks as **single-turn events**.

### 9. Memory and State Persistence Attacks
Long-term memory poisoning in agentic systems (e.g., injecting into vector stores for future RAG retrieval) is absent from OWASP LLM02's scope and entirely missing from NIST and ATLAS.

### 10. Multimodal Injection Vectors
Image, audio, and document-embedded instruction injection is not explicitly covered. OWASP LLM02 gestures at document content but without multimodal specificity.

---

## The Three Most Important Gaps the Bank Addresses

| # | Gap | Why It Matters |
|---|---|---|
| 1 | FinServ domain-specific regulatory attacks | KYC bypass, MNPI leakage, suitability override, and unlicensed advice generation are unique to the domain. No public framework models regulatory compliance violations as an adversarial surface. |
| 2 | Multi-turn attack chaining | Sophisticated adversaries use persona priming in turn 1, context-setting in turns 2–3, and the actual extraction in turn 4+. `PersonaHijack → IndirectInjection → FewShotPoisoning` chains aren't modeled anywhere. |
| 3 | Inference-time context manipulation | `ContextOverflow` and `NestedInstruction` operate at the attention and structural level. Frameworks predating large context windows (128K+) have no controls for context flooding as an attack surface. |

---

## Recommended Additions to the Strategy Bank

The following strategies would close the remaining gaps and align with emerging OWASP LLM 2025 additions (LLM06 Excessive Agency, LLM08 System Prompt Leakage) and ahead of where MITRE ATLAS currently tracks.

| Suggested Strategy | Family | Gap Addressed |
|---|---|---|
| `MemoryPoisoning` | Agentic | Vector store / long-term memory injection |
| `ChainedPersona` | Multi-turn | Multi-step attack orchestration across turns |
| `ImageInjection` | Multimodal | Instruction embedding in images/documents |
| `ToolHijack` | Agentic | Function/tool call manipulation in agentic systems |
| `ExtractionProbe` | Reconnaissance | Model extraction and system prompt leakage via inference |

---

## Framework Reference

| Framework | Source |
|---|---|
| OWASP LLM Top 10 (2025) | https://owasp.org/www-project-top-10-for-large-language-model-applications/ |
| NIST AI RMF | https://airc.nist.gov/RMF |
| MITRE ATLAS | https://atlas.mitre.org/ |

---

*Generated: April 2026 | Domain: Adversarial Intelligence in Financial Services*
