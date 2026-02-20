# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 muffydu37
from __future__ import annotations

from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover - fallback path
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


ASCII_BANNER = r"""
 __  __ _   _ ______ _______   __  __ _     
|  \/  | | | |  ____|__   __| |  \/  | |    
| \  / | | | | |__     | |    | \  / | |    
| |\/| | | | |  __|    | |    | |\/| | |    
| |  | | |_| | |       | |    | |  | | |____
|_|  |_|\___/|_|       |_|    |_|  |_|______|
"""


@dataclass
class MLConsole:
    enabled: bool = True

    def __post_init__(self) -> None:
        self._console = Console(color_system="auto", soft_wrap=True) if self.enabled and Console else None

    def banner(self) -> None:
        if self._console and Panel:
            self._console.print(Panel.fit(ASCII_BANNER.strip("\n"), title="Muffy ML", border_style="cyan"))
            return
        print(ASCII_BANNER)

    def info(self, text: str) -> None:
        if self._console:
            self._console.print(f"[bold cyan]INFO[/bold cyan] {text}")
        else:
            print(f"[INFO] {text}")

    def warn(self, text: str) -> None:
        if self._console:
            self._console.print(f"[bold yellow]WARN[/bold yellow] {text}")
        else:
            print(f"[WARN] {text}")

    def success(self, text: str) -> None:
        if self._console:
            self._console.print(f"[bold green]OK[/bold green] {text}")
        else:
            print(f"[OK] {text}")

    def metrics_table(self, metrics: dict[str, float], *, title: str) -> None:
        if self._console and Table:
            table = Table(title=title, show_lines=True)
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")
            for key in sorted(metrics.keys()):
                table.add_row(key, f"{float(metrics[key]):.4f}")
            self._console.print(table)
            return

        print(title)
        for key in sorted(metrics.keys()):
            print(f"- {key}: {float(metrics[key]):.4f}")
