from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from rengongtong.brain import Brain
from rengongtong.persona import PersonaWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("rengongtong")

app = typer.Typer(
    name="rengongtong",
    help="Réngōng tóng (人工童) — a local, persistent Agentic Tamagotchi",
    no_args_is_help=True,
)
console = Console()

PERSONA = PersonaWrapper()


def _load_or_create_brain(soul_path: Path | None) -> Brain:
    brain = Brain()
    if soul_path and soul_path.exists():
        brain.load_soul(soul_path)
    return brain


@app.command()
def feed(
    text: str = typer.Argument(..., help="Text to train on"),
    steps: int = typer.Option(3, "--steps", "-s", help="Number of training steps"),
    soul: Path = typer.Option(None, "--soul", "-S", help="Path to a soul snapshot to load first"),
) -> None:
    """Feed the entity — incremental LoRA fine-tuning on provided text."""
    brain = _load_or_create_brain(soul)
    report = brain.feed(text, steps=steps)
    console.print(Panel(f"[bold]Feed complete[/]\n"
                        f"Loss: {report.loss:.4f}\n"
                        f"Perplexity: {report.perplexity_after:.2f}\n"
                        f"Duration: {report.duration_seconds:.2f}s",
                        title="Training Report"))
    brain.save_soul()


@app.command()
def chat(
    message: str = typer.Argument(..., help="Your message to the entity"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m", help="Max response tokens"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    soul: Path = typer.Option(None, "--soul", "-S", help="Path to a soul snapshot"),
) -> None:
    """Chat with the entity."""
    brain = _load_or_create_brain(soul)
    response = brain.chat(message, max_new_tokens=max_tokens, temperature=temperature)
    console.print(Panel(response, title=f"[bold]{brain.mood}[/] Réngōng tóng"))


@app.command()
def status(
    soul: Path = typer.Option(None, "--soul", "-S", help="Path to a soul snapshot"),
) -> None:
    """Show the entity's current state."""
    brain = _load_or_create_brain(soul)
    s = brain.state

    table = Table(title=f"Réngōng tóng — {s.soul_id}")
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Age", f"{s.age_hours:.1f} hours ({s.age_hours / 24:.2f} days)")
    table.add_row("Mood", str(s.mood))
    table.add_row("Total feeds", str(s.total_feeds))
    table.add_row("Total dreams", str(s.total_dreams))
    table.add_row("Total decays", str(s.total_decays))
    table.add_row("Total tokens seen", f"{s.total_tokens_seen:,}")
    table.add_row("High attention", str(s.high_attention_mode))
    table.add_row("Curiosity level", f"{s.curiosity_level:.3f}")
    table.add_row("Last feed", str(s.last_feed or "never"))
    table.add_row("Last dream", str(s.last_dream or "never"))
    table.add_row("Last decay", str(s.last_decay or "never"))

    traits = s.personality_traits
    table.add_row("Franconian grumpiness", f"{traits['franconian_grumpiness']:.2f}")
    table.add_row("Scholarly humility", f"{traits['scholarly_humility']:.2f}")
    table.add_row("Curiosity drive", f"{traits['curiosity_drive']:.2f}")
    table.add_row("Identity stability", f"{traits['identity_stability']:.2f}")
    table.add_row("Forgetting rate", f"{traits['forgetting_rate']:.2f}")
    table.add_row("Current soul", s.current_soul_path or "none")

    console.print(table)


@app.command()
def dream(
    soul: Path = typer.Option(None, "--soul", "-S", help="Path to a soul snapshot"),
) -> None:
    """Trigger an immediate consolidation (dreaming) cycle."""
    brain = _load_or_create_brain(soul)
    from rengongtong.dreaming import ConsolidationRoutine
    routine = ConsolidationRoutine(brain.model, brain.tokenizer)
    report = routine.dream()
    brain.state.total_dreams += 1
    brain.state.last_dream = report.timestamp
    brain.save_soul()
    console.print(Panel(f"[bold]Dream complete[/]\n"
                        f"Steps: {report.steps}\n"
                        f"Loss: {report.loss:.4f}\n"
                        f"Duration: {report.duration_seconds:.2f}s",
                        title="Consolidation Report"))


@app.command()
def save(
    path: Path = typer.Argument(None, help="Output directory for the soul snapshot"),
    soul: Path = typer.Option(None, "--soul", "-S", help="Current soul path"),
) -> None:
    """Save a soul snapshot."""
    brain = _load_or_create_brain(soul)
    out = brain.save_soul(path)
    console.print(f"[green]Soul saved to[/] {out}")


@app.command()
def list_souls(
    base: Path = typer.Option(Path("souls"), "--base", "-b", help="Souls base directory"),
) -> None:
    """List all saved soul snapshots."""
    if not base.exists():
        console.print("[yellow]No souls directory found.[/]")
        return

    dirs = sorted(base.iterdir())
    if not dirs:
        console.print("[yellow]No soul snapshots found.[/]")

    table = Table(title="Soul Snapshots")
    table.add_column("Date", style="cyan")
    table.add_column("Path", style="green")
    for d in dirs:
        if d.is_dir():
            state_file = d / "state.json"
            date = d.name.split("--")[-1] if "--" in d.name else "unknown"
            if state_file.exists():
                data = json.loads(state_file.read_text())
                date = data.get("last_feed", date)
            table.add_row(str(date)[:19], str(d))

    console.print(table)


@app.command()
def run(
    soul: Path = typer.Option(None, "--soul", "-S", help="Initial soul snapshot"),
) -> None:
    """Run the full interactive entity loop (lifecycle + chat)."""
    brain = _load_or_create_brain(soul)
    console.print("[bold cyan]Réngōng tóng[/] is awake. Type your messages or commands.")

    async def _loop() -> None:
        async with brain.running():
            while True:
                try:
                    msg = console.input("[bold]You >[/] ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[yellow]Goodbye![/]")
                    brain.save_soul()
                    break

                if msg.lower() in ("/quit", "/exit", "/q"):
                    brain.save_soul()
                    break
                if msg.lower() == "/status":
                    for line in brain.state.model_dump().items():
                        console.print(f"  [cyan]{line[0]}[/]: {line[1]}")
                    continue
                if msg.lower().startswith("/feed "):
                    text = msg[6:]
                    report = brain.feed(text)
                    console.print(f"[dim]Learned: loss={report.loss:.4f}[/]")
                    continue

                response = brain.chat(msg)
                console.print(Panel(response, title=f"[bold]{brain.mood}[/] Réngōng tóng"))

    asyncio.run(_loop())


if __name__ == "__main__":
    app()
