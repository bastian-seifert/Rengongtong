# Réngōng tóng (人工童)

A local, persistent "Agentic Tamagotchi" built on **SmolLM2-1.7B** that evolves via
incremental LoRA updates, curiosity-driven exploration, biological weight decay,
experience replay, and **Matrix Perturbation Theory (MPT)** constructs for spectral
stability.

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
    ├── _types.py                   # MptConfig, ReplayConfig, DecayMode, Protocols
    ├── _spectral.py                # Pure MPT tensor math (Gershgorin, Davis-Kahan, Pseudospectral)
    ├── _jinja.py                   # Jinja2 template rendering engine
    ├── memory.py                   # ExperienceReplayBuffer — NLL & distillation replay
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
| **ExperienceReplayBuffer** | `memory.py` | Short-term episodic memory with NLL & distillation replay |
| **Spectral Math** | `_spectral.py` | Pure tensor functions for all MPT constructs |

### Design

- **Base model**: `HuggingFaceTB/SmolLM2-1.7B` (frozen, 4-bit quantized via Unsloth)
- **Soul**: LoRA adapter (`r=16`, `α=32`) on all linear layers
- **Memory**: Revolving buffer of daily soul snapshots in `souls/`
- **Experience Replay**: Short-term episodic buffer (FIFO/importance sampling, NLL/distillation modes)
- **Persona**: 6 mood states (grantig, scholarly, curious, bored, hungry, neutral) with Franconian vocabulary injection

---

## Matrix Perturbation Theory (MPT)

Three MPT-inspired constructs regularize the entity's "soul".  Read the white
paper's erratum (`white_paper/results/experiment_001/paper.md`) before relying on
the original claims: in practice the Gershgorin decay acts as norm-dependent
shrinkage of the merged LoRA update, and its benefit over *shrink-matched*
baselines is still being re-measured (`experiment_002`).

### 1. Gershgorin Circle Decay (`metabolism.py`)

Penalizes off-diagonal mass of the merged update. The Gershgorin radius `R_i = Σ_{j≠i} |w_ij|` is computed on
the **merged `B@A` weight update** (square attention projections). Rows where
`R_i > |w_ii|` get an extra decay penalty on their off-diagonal elements.

**Modes** (configurable via `MptConfig.decay_mode`):
- `saliency` — magnitude-based protection (original behavior)
- `gershgorin` — penalizes off-diagonal elements in unstable rows of merged `B@A`
- `hybrid` — both saliency protection and Gershgorin off-diagonal penalties
- `diagonal_mass` / `rotated_gershgorin` — mechanism controls
- `uniform` — plain shrinkage `w ← (1 − λ(t)) w`, the null-hypothesis baseline
- `none` — no decay

### 2. Perturbation Sensitivity (`curiosity.py`, `_spectral.py`)

Historically called "pseudospectral stability", but it is a random-perturbation
*sharpness* measure, not a resolvent norm: inject ε-scale noise into all LoRA
weights, measure `‖logits − logits_perturbed‖ / ‖logits‖`, then restore the
weights exactly. A large value means small weight changes cause large output
swings.

### 3. Subspace Protection (`dreaming.py`, `_spectral.py`)

During the dream cycle, penalizes the fraction of each LoRA weight's energy that
leaves the principal right-singular subspace captured at the start of the cycle
(a differentiable surrogate of the Davis–Kahan sin Θ angle):

```
Loss_total = Loss_NLL + λ · mean_layers ‖W (I − V_ref V_refᵀ)‖²_F / ‖W‖²_F
```

The reference is the LoRA adapter itself, not the pretrained weights W₀. To
measure drift away from the pretrained foundation, use
`_spectral.intruder_dimensions(W0, W0 + ΔW)` (Shuttleworth et al. 2024).

---

## Experience Replay Buffer (`memory.py`)

A short-term episodic memory that replays recent inputs during the feed cycle
to complement MPT spectral constraints:

- **NLL Replay** (default): replayed texts are concatenated with new data and
  trained with standard cross-entropy loss
- **Distillation Replay**: the model's own logits are cached at push time; on
  replay the loss is `KL(cached_logits ‖ current_logits)`, preserving "dark
  knowledge"

### Integration

```
User feeds "text A"
  → push("text A") stores it in the buffer
  → sample(replay_ratio × buffer_size) retrieves old experiences
  → extra optimizer step on replay loss
  → Gershgorin decay protects against eigenvalue drift from replay
  → Davis-Kahan during dreaming keeps subspaces aligned
```

The buffer is persisted in `state.json` alongside the entity state and restored
on `load_soul`.

## Usage

### CLI Commands

```bash
# Basic feed (incremental LoRA fine-tuning)
rengongtong feed "Your text here" --steps 5

# Feed with experience replay buffer (capacity 200, NLL mode)
rengongtong feed "Your text here" --replay-capacity 200 --replay-mode nll

# Feed with distillation replay
rengongtong feed "Your text here" -R 200 --replay-mode distill

# Chat with the entity
rengongtong chat "Hello, how are you?"

# Check status (age, mood, stability, token count, replay count)
rengongtong status

# Trigger a dream cycle (self-distillation)
rengongtong dream

# Save a soul snapshot explicitly
rengongtong save

# List all saved souls
rengongtong list-souls

# Interactive mode with full lifecycle and replay
rengongtong run -R 200 --replay-mode nll
```

### Feeding Best Practices

- **Use moderate steps** (3–10). SmolLM2-1.7B is small — 50+ steps of repetitive
  text can overwhelm the LoRA adapters even with MPT protection.
- **Vary your text**. Multiple short feeds are better than one long repetitive one.
- **Enable replay** with `--replay-capacity 200` to consolidate old knowledge
  alongside new data without extra feeds.
- **Watch perplexity** via `status`. If it drops below 3.0, the entity is bored
  and needs more diverse input.

### Understanding MPT Metrics

The `status` command shows `stability_gap` in the entity state. This is the
normalized perturbation sensitivity (see `CuriosityController.measure_stability`).
Nothing calls it automatically yet, so the value stays 0 until you do:

| Value | Meaning |
|---|---|
| `< 0.02` | Very stable — safe learning zone |
| `0.02–0.05` | Moderate — some perturbation sensitivity |
| `> 0.05` | Unstable — high output sensitivity to weight noise |

These thresholds are heuristics; whether sensitivity predicts hallucination in
this setup has not been measured.

### Interactive Mode (`run`)

The `run` command starts an interactive REPL with a background metabolic loop.
Pass `--replay-capacity` to enable the experience replay buffer:

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
from rengongtong._types import MptConfig, DecayMode, ReplayConfig

mpt = MptConfig(
    decay_mode=DecayMode.HYBRID,              # Gershgorin + saliency
    gershgorin_penalty=0.05,                  # gentle constraint on merged B@A
    pseudospectral_epsilon=1e-5,              # noise scale for stability check
    pseudospectral_threshold=0.05,            # sensitivity above which mood becomes "bored"
    subspace_protection_weight=0.05,          # subspace-leakage loss weight during dreaming
    subspace_rank=8,                          # number of singular vectors to align
)

replay = ReplayConfig(
    mode="nll",                               # nll | distill
    strategy="fifo",                          # fifo | importance
    capacity=200,
    replay_ratio=0.3,
)

brain = Brain(mpt=mpt, replay=replay)
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
counters, **replay buffer state**, and **stability metrics** — the complete
"state of mind" at snapshot time.

Load a previous soul:
```bash
rengongtong chat "Wos host du glernt?" --soul souls/a1b2c3.../
```

---

## Development

```bash
uv sync --dev           # includes pytest, ruff, mypy
uv run pytest
uv run ruff check .     # linting
uv run mypy rengongtong  # type checking
```

## License

MIT
