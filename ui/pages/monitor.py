"""Monitor page — agent status, memory, and registered tool inventory."""

from __future__ import annotations

import json
import time

import flet as ft

from ui import api_client
from ui.theme import T


TOOL_ARG_TEMPLATES = {
    "calculator": {"expression": "25 * 100"},
    "datetime": {"format_type": "current", "value": None},
    "sec_search": {"ticker": "NVDA", "form_type": "10-K", "limit": 5},
    "sec_extract": {"ticker": "NVDA", "form_type": "10-K", "section": "item1a"},
    "knowledge_search": {"query": "revenue growth drivers", "k": 5},
    "send_message": {"user_id": "finnav_test_user", "message": "hello from playground", "channel": "voiceflow"},
    "send_alert": {
        "user_id": "finnav_test_user",
        "alert_type": "price",
        "title": "NVDA up",
        "message": "NVDA crossed your threshold",
        "priority": "normal",
    },
}


def build(page: ft.Page) -> ft.Control:
    backend_text = ft.Text("…", size=12, color=T.TEXT_DIM)
    metrics_row = ft.Row(spacing=0)
    tools_block = ft.Column(tight=True, spacing=T.S3)
    raw = ft.Text("", size=11, font_family="JetBrains Mono, Consolas, monospace", color=T.TEXT_DIM, selectable=True)
    spinner = ft.ProgressRing(width=14, height=14, visible=False, stroke_width=2, color=T.ACCENT)
    tool_name_dd = ft.Dropdown(
        label="Tool",
        width=260,
        text_size=12,
        bgcolor=T.SURFACE,
        border_color=T.BORDER,
        focused_border_color=T.ACCENT,
        color=T.TEXT,
        options=[],
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
    )
    tool_args = ft.TextField(
        label="Args (JSON)",
        value=json.dumps({}, indent=2),
        multiline=True,
        min_lines=7,
        max_lines=12,
        text_size=12,
        text_style=ft.TextStyle(font_family="JetBrains Mono, Consolas, monospace"),
        border_color=T.BORDER,
        focused_border_color=T.ACCENT,
        bgcolor=T.SURFACE,
        color=T.TEXT,
        content_padding=ft.Padding.all(T.S3),
    )
    run_latency = ft.Text("—", size=11, color=T.TEXT_DIM, font_family="JetBrains Mono, Consolas, monospace")
    tool_output = ft.Text("", size=11, color=T.TEXT_DIM, selectable=True,
                          font_family="JetBrains Mono, Consolas, monospace")

    def set_tool_args_template(name: str | None) -> None:
        template = TOOL_ARG_TEMPLATES.get(name or "", {})
        tool_args.value = json.dumps(template, indent=2)

    def on_tool_change(e: ft.ControlEvent) -> None:
        set_tool_args_template(e.control.value)
        page.update()

    tool_name_dd.on_change = on_tool_change

    def stat(label: str, value: str) -> ft.Control:
        return ft.Container(
            expand=True,
            padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
            border=ft.Border.only(right=ft.BorderSide(1, T.BORDER)),
            content=ft.Column([
                ft.Text(label.upper(), size=9, color=T.TEXT_FAINT, weight=ft.FontWeight.W_600),
                ft.Text(value, size=20, color=T.TEXT, weight=ft.FontWeight.W_600),
            ], tight=True, spacing=T.S1),
        )

    def tool_chip(name: str, owners: list[str]) -> ft.Control:
        owner_text = ", ".join(owners) if owners else "shared"
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R_SM,
            bgcolor=T.SURFACE,
            content=ft.Row([
                ft.Text(name, size=11, color=T.TEXT, weight=ft.FontWeight.W_600,
                        font_family="JetBrains Mono, Consolas, monospace"),
                ft.Container(width=1, height=10, bgcolor=T.BORDER),
                ft.Text(owner_text, size=10, color=T.TEXT_DIM),
            ], spacing=T.S2, tight=True),
        )

    async def refresh(_e=None) -> None:
        spinner.visible = True
        page.update()
        try:
            health = await api_client.health()
            status = await api_client.team_status()
            tools = await api_client.list_tools()
            mem = status.get("memory", {})
            backend_text.value = (
                f"backend = {health.get('backend')}  ·  ready = {health.get('team_ready')}"
            )
            metrics_row.controls = [
                stat("Agents",   str(status.get("registered_agents", 0))),
                stat("Tools",    str(len(tools.get("tools", [])))),
                stat("Memories", str(mem.get("total_memories", 0))),
                stat("Turns",    str(mem.get("conversation_turns", 0))),
                stat("Facts",    str(mem.get("facts", 0))),
            ]
            if metrics_row.controls:
                metrics_row.controls[-1].border = None

            # Tools panel: group by agent
            by_agent = tools.get("by_agent", {})
            tool_meta = {t["name"]: t for t in tools.get("tools", [])}
            blocks: list[ft.Control] = []
            for agent_name, tool_names in by_agent.items():
                if not tool_names:
                    continue
                chips = [tool_chip(n, tool_meta.get(n, {}).get("owners", [])) for n in tool_names]
                blocks.append(ft.Column([
                    ft.Text(agent_name.upper(), size=10, color=T.TEXT_FAINT, weight=ft.FontWeight.W_600),
                    ft.Row(chips, wrap=True, spacing=T.S2, run_spacing=T.S2),
                ], tight=True, spacing=T.S2))
            tools_block.controls = blocks or [ft.Text("No tools registered.", size=11, color=T.TEXT_FAINT)]
            tool_names = [t.get("name", "") for t in tools.get("tools", []) if t.get("name")]
            tool_name_dd.options = [ft.dropdown.Option(n) for n in tool_names]
            if tool_names:
                if tool_name_dd.value not in tool_names:
                    tool_name_dd.value = tool_names[0]
                    set_tool_args_template(tool_name_dd.value)
            else:
                tool_name_dd.value = None
                set_tool_args_template(None)

            raw.value = json.dumps(status, indent=2)[:6000]
        except Exception as ex:
            raw.value = f"Error: {ex}"
        finally:
            spinner.visible = False
            page.update()

    async def run_selected_tool(_e=None) -> None:
        name = tool_name_dd.value
        if not name:
            tool_output.value = "No tool selected."
            run_latency.value = "—"
            page.update()
            return

        raw_args = (tool_args.value or "").strip() or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as ex:
            tool_output.value = f"Invalid JSON args: {ex}"
            run_latency.value = "—"
            page.update()
            return
        if not isinstance(args, dict):
            tool_output.value = "Args JSON must be an object."
            run_latency.value = "—"
            page.update()
            return

        start = time.perf_counter()
        try:
            result = await api_client.run_tool(name, args)
            elapsed_ms = (time.perf_counter() - start) * 1000
            run_latency.value = f"{elapsed_ms:.0f} ms"

            if isinstance(result, dict):
                if result.get("ok") is False and "error" in result:
                    output_str = str(result.get("error", ""))
                else:
                    output_str = str(result.get("result", result))
            else:
                output_str = str(result)

            try:
                parsed = json.loads(output_str)
                tool_output.value = json.dumps(parsed, indent=2)
            except Exception:
                tool_output.value = output_str
        except Exception as ex:
            elapsed_ms = (time.perf_counter() - start) * 1000
            run_latency.value = f"{elapsed_ms:.0f} ms"
            tool_output.value = f"Error: {ex}"
        finally:
            page.update()

    page.run_task(refresh)

    def panel(title: str, body: ft.Control, expand: bool = False) -> ft.Control:
        return ft.Container(
            bgcolor=T.SURFACE,
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            expand=expand,
            content=ft.Column([
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
                    border=ft.Border.only(bottom=ft.BorderSide(1, T.BORDER)),
                    content=ft.Text(title, size=11, color=T.TEXT_DIM, weight=ft.FontWeight.W_600),
                ),
                ft.Container(padding=ft.Padding.all(T.S4), content=body),
            ], tight=True, spacing=0),
        )

    return ft.Column([
        ft.Row([
            ft.Text("Monitor", size=18, weight=ft.FontWeight.W_600, color=T.TEXT),
            ft.Container(width=1, height=14, bgcolor=T.BORDER),
            backend_text,
            ft.Container(expand=True),
            ft.IconButton(icon=ft.Icons.REFRESH, icon_size=16, icon_color=T.TEXT_DIM,
                          on_click=refresh, tooltip="Refresh"),
            spinner,
        ], spacing=T.S3),
        ft.Container(
            bgcolor=T.SURFACE, border=ft.Border.all(1, T.BORDER), border_radius=T.R,
            content=metrics_row,
        ),
        panel("Registered tools", tools_block),
        panel("Run a tool", ft.Column([
            ft.Row([
                tool_name_dd,
                ft.ElevatedButton(
                    "Run",
                    icon=ft.Icons.PLAY_ARROW,
                    on_click=run_selected_tool,
                    style=ft.ButtonStyle(
                        bgcolor=T.ACCENT,
                        color="#FFFFFF",
                        shape=ft.RoundedRectangleBorder(radius=T.R),
                        padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
                    ),
                ),
                ft.Text("Latency", size=11, color=T.TEXT_DIM),
                run_latency,
            ], spacing=T.S3, wrap=True),
            tool_args,
            ft.Container(
                padding=ft.Padding.all(T.S3),
                border=ft.Border.all(1, T.BORDER_FAINT),
                border_radius=T.R_SM,
                bgcolor=T.SURFACE,
                content=tool_output,
            ),
        ], spacing=T.S3, tight=True)),
        ft.ExpansionTile(
            title=ft.Text("Raw team status", size=12, color=T.TEXT_DIM),
            controls=[
                ft.Container(
                    padding=ft.Padding.all(T.S4),
                    border=ft.Border.all(1, T.BORDER_FAINT),
                    border_radius=T.R_SM,
                    bgcolor=T.SURFACE,
                    content=raw,
                ),
            ],
        ),
    ], spacing=T.S4, expand=True, scroll=ft.ScrollMode.AUTO)
