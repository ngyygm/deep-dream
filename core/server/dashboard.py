"""
DeepDream Dashboard — 基于 rich 的实时终端监控仪表盘。

层级布局：
  顶部: 系统总览条（运行时间 | 图谱数 | 线程数 | 模式）
  中部左: 图谱列表（Tree 层级展示，含存储/队列摘要）
  中部右: 活跃任务（进度条 + 百分比）
  底部左: 队列 & API 统计
  底部右: 最近系统日志
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from core.server.monitor import (
    LOG_MODE_MONITOR,
    SystemMonitor,
    _format_seconds,
)


class DeepDreamDashboard:
    """基于 rich.Live 的实时终端仪表盘，从 SystemMonitor 读取数据。"""

    def __init__(
        self,
        monitor: SystemMonitor,
        refresh_interval: float = 1.0,
    ):
        self.monitor = monitor
        self.refresh_interval = max(0.3, refresh_interval)
        self._live: Optional[Live] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """在独立线程中启动 rich.Live 仪表盘。"""
        self._thread = threading.Thread(target=self._run, name="tmg-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止仪表盘。"""
        self._stop.set()
        if self._live:
            try:
                self._live.stop()
            except Exception as _e:
                logging.getLogger(__name__).debug("停止仪表盘 Live 失败: %s", _e)

    def _run(self) -> None:
        """仪表盘主循环。"""
        with Live(
            self._build_layout(),
            refresh_per_second=max(1, 1.0 / self.refresh_interval),
            screen=True,
            transient=False,
        ) as live:
            self._live = live
            while not self._stop.is_set():
                live.update(self._build_layout())
                self._stop.wait(self.refresh_interval)
            self._live = None

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="graphs", ratio=3),
            Layout(name="stats", ratio=2),
        )
        layout["right"].split_column(
            Layout(name="tasks", ratio=3),
            Layout(name="logs", ratio=2),
        )
        layout["header"].update(self._render_header())
        layout["graphs"].update(self._render_graphs())
        layout["tasks"].update(self._render_tasks())
        layout["stats"].update(self._render_stats())
        layout["logs"].update(self._render_logs())
        layout["footer"].update(self._render_footer())
        return layout

    # ------------------------------------------------------------------
    # 各区块渲染
    # ------------------------------------------------------------------

    def _render_header(self) -> Panel:
        overview = self.monitor.overview()
        header_text = Text()
        header_text.append("DeepDream", style="bold cyan")
        header_text.append("  System Monitor", style="bold white")
        header_text.append("  |  Uptime: ", style="dim")
        header_text.append(overview["uptime_display"], style="green")
        header_text.append("  |  Graphs: ", style="dim")
        header_text.append(str(overview["graph_count"]), style="yellow")
        header_text.append("  |  Threads: ", style="dim")
        header_text.append(str(overview["python_threads_total"]), style="magenta")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        header_text.append(f"  |  {now}", style="dim")
        return Panel(header_text, style="bold blue")

    def _render_graphs(self) -> Panel:
        graphs = self.monitor.all_graphs()
        if not graphs:
            return Panel("[dim]No graphs[/dim]", title="[bold]Graphs[/bold]", border_style="blue")

        tree = Tree("[bold cyan]Graphs[/bold cyan]")
        for g in graphs:
            gid = g["graph_id"]
            s = g["storage"]
            q = g["queue"]
            label = f"[bold]{gid}[/bold]"
            stats_line = (
                f"  [dim]E:{s['entities']}[/dim] "
                f"[dim]R:{s['relations']}[/dim] "
                f"[dim]C:{s['episodes']}[/dim]"
            )
            if q["running_count"] > 0:
                stats_line += f"  [green]Running:{q['running_count']}[/green]"
            if q["queued_count"] > 0:
                stats_line += f"  [yellow]Queued:{q['queued_count']}[/yellow]"
            branch = tree.add(f"{label}{stats_line}")
            # 显示活跃任务
            for task in g.get("queue", {}).get("active_tasks", []):
                tid = task.get("task_id", "?")[:8]
                sn = task.get("source_name", "?")[:20]
                progress = task.get("progress", 0.0)
                phase = task.get("phase_label", "")
                pct = f"{progress * 100:.0f}%"
                bar_len = 10
                filled = int(round(progress * bar_len))
                bar = "█" * filled + "░" * max(0, bar_len - filled)
                task_line = f"[dim]{tid}[/dim] {sn} [{bar}] {pct}"
                if phase:
                    task_line += f" [cyan]{phase}[/cyan]"
                branch.add(task_line)

        return Panel(tree, title="[bold]Graphs[/bold]", border_style="blue")

    def _render_tasks(self) -> Panel:
        tasks = self.monitor.all_tasks(limit=20)
        active = [t for t in tasks if t["status"] in ("running", "queued")]

        if not active:
            return Panel(
                "[dim]No active tasks[/dim]",
                title="[bold]Active Tasks[/bold]",
                border_style="green",
            )

        now = time.time()
        lines: list[str] = []
        for task in active:
            tid = task.get("task_id", "?")[:8]
            sn = task.get("source_name", "?")[:20]
            status = task["status"]
            progress = task.get("progress", 0.0)
            phase = task.get("phase_label", "")

            # --- 耗时 ---
            started_at = task.get("started_at")
            created_at = task.get("created_at", now)
            elapsed = 0.0
            if status == "running" and started_at:
                elapsed = now - started_at
            elif status == "queued":
                elapsed = now - created_at
            elapsed_str = _format_seconds(elapsed)

            # --- 进度条（固定宽度 ASCII） ---
            bar_len = 16
            filled = int(round(progress * bar_len))
            bar = "\u2588" * filled + "\u2591" * max(0, bar_len - filled)
            pct = f"{progress * 100:5.1f}%"

            # --- 状态颜色 ---
            if status == "running":
                status_tag = "[green]\u25cf[/green]"
                phase_tag = f"[cyan]{phase}[/cyan]"
            else:
                status_tag = "[yellow]\u25cb[/yellow]"
                phase_tag = f"[yellow]{phase}[/yellow]"

            line = (
                f"{status_tag} [{tid}] {sn}\n"
                f"    [{bar}] {pct}  {phase_tag}  [dim]{elapsed_str}[/dim]"
            )
            lines.append(line)

        content = "\n\n".join(lines)
        return Panel(
            content,
            title=f"[bold]Active Tasks ({len(active)})[/bold]",
            border_style="green",
        )

    def _render_stats(self) -> Panel:
        # 合并所有图谱的队列统计 + API 访问统计
        graphs = self.monitor.all_graphs()
        total_queued = sum(g["queue"]["queued_count"] for g in graphs)
        total_running = sum(g["queue"]["running_count"] for g in graphs)
        total_backlog = sum(g["queue"]["backlog"] for g in graphs)

        access = self.monitor.access_stats(since_seconds=300)

        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Key", style="bold cyan", width=16)
        table.add_column("Value", style="white")

        table.add_row("Queued", f"[yellow]{total_queued}[/yellow]")
        table.add_row("Running", f"[green]{total_running}[/green]")
        table.add_row("Backlog", f"[dim]{total_backlog}[/dim]")

        table.add_row("", "")
        table.add_row("API Requests", f"[white]{access['total_requests']}[/white]")
        table.add_row("Errors", f"[red]{access['error_count']}[/red]" if access["error_count"] > 0 else "[green]0[/green]")
        table.add_row("Success Rate", f"[green]{access['success_rate']}%[/green]")
        table.add_row("Avg Latency", f"{access['avg_duration_ms']}ms")
        table.add_row("Req/min", f"[magenta]{access['requests_per_minute']}[/magenta]")

        return Panel(table, title="[bold]Queue & API[/bold]", border_style="yellow")

    def _render_logs(self) -> Panel:
        logs = self.monitor.recent_logs(limit=12)

        if not logs:
            return Panel(
                "[dim]No recent logs[/dim]",
                title="[bold]System Logs[/bold]",
                border_style="red",
            )

        lines = []
        for log in logs[:12]:
            ts = log.get("time", "")
            level = log.get("level", "INFO")
            source = log.get("source", "")
            message = log.get("message", "")

            if level == "ERROR":
                style = "bold red"
            elif level == "WARN":
                style = "yellow"
            else:
                style = "dim"

            line = f"[{style}]{ts} [{source}] {message}[/{style}]"
            lines.append(line)

        log_text = "\n".join(lines)
        return Panel(
            log_text,
            title="[bold]System Logs[/bold]",
            border_style="red",
        )

    def _render_footer(self) -> Panel:
        footer = Text()
        footer.append(" Press Ctrl+C to stop ", style="bold on black")
        footer.append(" | ", style="dim")
        footer.append(f"Refresh: {self.refresh_interval}s", style="dim")
        footer.append(" | ", style="dim")
        footer.append("API: GET /api/system/*", style="dim cyan")
        return Panel(footer, style="dim")
