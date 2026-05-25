from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent / "templates"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    keep_trailing_newline=True,
)


def render_template(name: str, **kwargs) -> str:
    return env.get_template(name).render(**kwargs)
