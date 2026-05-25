# Réngōng tóng (人工童)

A local, persistent "Agentic Tamagotchi" built on SmolLM2-135M that evolves via incremental LoRA updates, curiosity-driven exploration, and biological weight decay.

> *"A fränkischa kloaner Bua mit am großen Wissensdurst."*

## Architecture

```
rengongtong/
├── pyproject.toml          # Project metadata & dependencies
├── souls/                  # LoRA adapter snapshots (gitignored)
└── rengongtong/
    ├── __init__.py
    ├── __main__.py         # python -m rengongtong
    ├── _state.py           # Pydantic models (EntityState, Mood, …)
    ├── _types.py           # Protocols & type aliases
    ├── brain.py            # Brain — central orchestrator
    ├── synaptic.py         # SynapticManager — 4-bit quant + LoRA soul
    ├── metabolism.py       # MetabolicLoop — saliency-preserving decay
    ├── dreaming.py         # ConsolidationRoutine — self-distillation
    ├── curiosity.py        # CuriosityController — perplexity drive
    ├── persona.py          # PersonaWrapper — Franconian-Scholar
    └── cli.py              # typer CLI
```

### Components

| Component | File | Role |
|---|---|---|
| **Brain** | `brain.py` | Central orchestrator — feed, chat, soul persistence, async lifecycle |
| **SynapticManager** | `synaptic.py` | 4-bit quantization (bitsandbytes), LoRA adapter buffer, merge/unload, mood scaling |
| **MetabolicLoop** | `metabolism.py` | Temporal weight decay with saliency preservation |
| **ConsolidationRoutine** | `dreaming.py` | Self-distillation: concept extraction → reflective Q&A → self-fine-tune |
| **CuriosityController** | `curiosity.py` | Perplexity measurement, Goldilocks-zone detection, proactive questioning |
| **PersonaWrapper** | `persona.py` | East Franconian dialect + Chinese scholarly humility injection |

### Design

- **Base model**: `HuggingFaceTB/SmolLM2-135M` (frozen, 4-bit quantized)
- **Soul**: LoRA adapter (r=16, α=32) on all linear layers
- **Memory**: Revolving buffer of daily soul snapshots in `souls/`
- **Decay**: λ(t) = λ₀·exp(−γ·Δt) with saliency-weighted protection
- **Dreaming**: Extracts top-k hidden-state concepts → generates relational questions → self-answers → fine-tunes
- **Curiosity**: Mid-range perplexity (3–20) → high-attention mode; low perplexity (<3) → proactive user question
- **Persona**: 6 mood states (grantig, scholarly, curious, bored, hungry, neutral) with Franconian vocabulary injection

## Installation

```bash
# Requires Python 3.12+ and CUDA GPU

# Recommended — uv (fast, reliable):
uv sync
uv run rengongtong status

# Or with pip:
pip install -e .
```

## Usage

### CLI Commands

```bash
# Feed the entity (incremental LoRA fine-tuning)
rengongtong feed "Your text here" --steps 5

# Chat interactively
rengongtong chat "Hello, how are you?"

# Show entity status
rengongtong status

# Trigger dream cycle
rengongtong dream

# Save soul snapshot
rengongtong save

# List all saved souls
rengongtong list-souls

# Interactive mode with full lifecycle
rengongtong run
```

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

The `state.json` captures the entity's age, mood, personality traits, and memory counters — the complete "state of mind" at snapshot time.

## License

MIT
