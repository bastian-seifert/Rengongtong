# Réngōng tóng (人工童)

A local, persistent "Agentic Tamagotchi" built on **SmolLM2-135M** that evolves via
incremental LoRA updates, curiosity-driven exploration, biological weight decay, and
**Matrix Perturbation Theory (MPT)** constructs for spectral stability.

> *"A fränkischa kloaner Bua mit am großen Wissensdurst."*

---

## Quick Start

```bash
uv sync                                    # install dependencies (Python 3.12+, CUDA GPU)
uv run rengongtong feed "Hello world" --steps 3   # teach it something
uv run rengongtong chat "Wia gehts da?"           # talk to it
uv run rengongtong status                         # check its state of mind
uv run rengongtong dream                          # let it consolidate overnight
```

---

## Architecture

```
rengongtong/
├── pyproject.toml
├── souls/                          # LoRA adapter snapshots (gitignored)
└── rengongtong/
    ├── __init__.py
    ├── __main__.py                 # python -m rengongtong
    ├── _state.py                   # Pydantic models (EntityState, Mood, reports)
    ├── _types.py                   # MptConfig, DecayMode, Protocols
    ├── _spectral.py                # Pure MPT tensor math (Gershgorin, Davis-Kahan, Pseudospectral)
    ├── _jinja.py                   # Jinja2 template rendering engine
    ├── brain.py                    # Brain — central orchestrator
    ├── synaptic.py                 # SynapticManager — 4-bit quant + LoRA soul
    ├── metabolism.py               # MetabolicLoop — saliency & Gershgorin decay
    ├── dreaming.py                 # ConsolidationRoutine — subspace-protected self-distillation
    ├── curiosity.py                # CuriosityController — perplexity + stability drive
    ├── persona.py                  # PersonaWrapper — Franconian-Scholar dialect
    └── cli.py                      # typer CLI
```

### Component Roles

| Component | File | Role |
|---|---|---|
| **Brain** | `brain.py` | Central orchestrator — feed, chat, soul persistence, async lifecycle |
| **SynapticManager** | `synaptic.py` | 4-bit quantization, LoRA adapter, merge/unload, mood-driven scaling |
| **MetabolicLoop** | `metabolism.py` | Temporal weight decay with saliency & Gershgorin spectral protection |
| **ConsolidationRoutine** | `dreaming.py` | Self-distillation with Davis-Kahan subspace alignment |
| **CuriosityController** | `curiosity.py` | Perplexity measurement + pseudospectral stability checking |
| **PersonaWrapper** | `persona.py` | East Franconian dialect + Chinese scholarly humility injection |
| **Spectral Math** | `_spectral.py` | Pure tensor functions for all MPT constructs |

### Design

- **Base model**: `HuggingFaceTB/SmolLM2-135M` (frozen, 4-bit quantized via Unsloth)
- **Soul**: LoRA adapter (`r=16`, `α=32`) on all linear layers
- **Memory**: Revolving buffer of daily soul snapshots in `souls/`
- **Persona**: 6 mood states (grantig, scholarly, curious, bored, hungry, neutral) with Franconian vocabulary injection

---

## Matrix Perturbation Theory (MPT)

Three MPT concepts mathematically guard the entity's "soul" against catastrophic
forgetting and eigenvalue drift:

### 1. Gershgorin Circle Decay (`metabolism.py`)

Prevents eigenvalues from wandering into the complex plane, which causes erratic
"mood swings." The Gershgorin radius `R_i = Σ_{j≠i} |w_ij|` is computed on
the **merged `B@A` weight update** (square attention projections). Rows where
`R_i > |w_ii|` get an extra decay penalty on their off-diagonal elements.

**Modes** (configurable via `MptConfig.decay_mode`):
- `saliency` — magnitude-based protection (original behavior)
- `gershgorin` — penalizes off-diagonal elements in unstable rows of merged `B@A`
- `hybrid` — both saliency protection and Gershgorin off-diagonal penalties

### 2. Pseudospectral Stability (`curiosity.py`, `_spectral.py`)

Measures the resolvent norm `‖(zI − W)^{-1}‖` via finite perturbation:
inject ε-scale noise into all LoRA weights, measure `‖logits − logits_perturbed‖ / ‖logits‖`.
A large gap signals a "danger zone" where small weight changes cause large
output swings — a precursor to hallucination.

### 3. Davis-Kahan Subspace Protection (`dreaming.py`, `_spectral.py`)

During the dream cycle, penalizes the sine of the angle between the principal
singular vectors of the base and current LoRA weights. This ensures the entity
learns new data without rotating its "fundamental logic" away from the pretrained
foundation.

```
Loss_total = Loss_NLL + λ · ‖sin Θ(V_base, V_soul)‖_F
```

---

## Usage

### CLI Commands

```bash
# Basic feed (incremental LoRA fine-tuning)
rengongtong feed "Your text here" --steps 5

# Chat with the entity
rengongtong chat "Hello, how are you?"

# Check status (age, mood, stability, token count)
rengongtong status

# Trigger a dream cycle (self-distillation)
rengongtong dream

# Save a soul snapshot explicitly
rengongtong save

# List all saved souls
rengongtong list-souls

# Interactive mode with full lifecycle
rengongtong run
```

### Feeding Best Practices

- **Use moderate steps** (3–10). SmolLM2-135M is small — 50+ steps of repetitive
  text can overwhelm the LoRA adapters even with MPT protection.
- **Vary your text**. Multiple short feeds are better than one long repetitive one.
- **Watch perplexity** via `status`. If it drops below 3.0, the entity is bored
  and needs more diverse input.

### Understanding MPT Metrics

The `status` command shows `stability_gap` in the entity state. This is the
normalized pseudospectral sensitivity:

| Value | Meaning |
|---|---|
| `< 0.02` | Very stable — safe learning zone |
| `0.02–0.05` | Moderate — some perturbation sensitivity |
| `> 0.05` | Unstable — high risk of hallucination/erratic behavior |

A rising stability gap during feeding indicates the new data is perturbing the
model's matrix structure. The Gershgorin and Davis-Kahan constraints
counteract this drift.

### Interactive Mode (`run`)

The `run` command starts an interactive REPL with a background metabolic loop:

```
Réngōng tóng is awake. Type your messages or commands.
You > Hallo, wia gehts da?
╭─────────────────────── grantig Réngōng tóng ────────────────────────╮
│ Ja mecha! I hob lang nix glernt heit. Wos konnst ma zoang?          │
╰─────────────────────────────────────────────────────────────────────╯
You > /feed Ein spannender Artikel über künstliche Intelligenz...
You > /status
You > /quit
```

**In-REPL commands:**
- `/feed <text>` — feed new text without leaving the session
- `/status` — print raw entity state
- `/quit`, `/exit`, `/q` — save and exit

### Python API — Custom MPT Configuration

```python
from rengongtong.brain import Brain
from rengongtong._types import MptConfig, DecayMode

mpt = MptConfig(
    decay_mode=DecayMode.HYBRID,              # Gershgorin + saliency
    gershgorin_penalty=0.05,                  # gentle constraint on merged B@A
    pseudospectral_epsilon=1e-5,              # noise scale for stability check
    pseudospectral_threshold=10.0,            # gap above which mood becomes "bored"
    subspace_protection_weight=0.05,          # Davis-Kahan loss weight during dreaming
    subspace_rank=8,                          # number of singular vectors to align
)

brain = Brain(mpt=mpt)
brain.feed("The sky is neon green.", steps=50)

from rengongtong.dreaming import ConsolidationRoutine
routine = ConsolidationRoutine(
    brain.model, brain.tokenizer,
    subspace_protection_weight=0.05,
    subspace_rank=8,
)
routine.capture_base_subspace()
routine.dream()
```

---

## Soul Snapshots

Every `save` or `run` exit creates a timestamped snapshot in `souls/`:

```
souls/
├── a1b2c3d4e5f6--20260525T120000/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── state.json
└── ...
```

The `state.json` captures the entity's age, mood, personality traits, memory
counters, and **stability metrics** — the complete "state of mind" at snapshot
time.

Load a previous soul:
```bash
rengongtong chat "Wos host du glernt?" --soul souls/a1b2c3.../
```

---

## Development

```bash
uv sync --dev           # includes pytest, ruff, mypy
uv run pytest           # 130+ tests
uv run ruff check .     # linting
uv run mypy rengongtong  # type checking
```

## License

MIT
