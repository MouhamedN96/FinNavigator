"""Schedules page — list / create / delete recurring research jobs."""

from __future__ import annotations

import flet as ft

from ui import api_client
from ui.theme import T


CRON_HINT = "min hour dom mon dow  ·  e.g.  0 9 * * 1  → every Mon 9am"


def build(page: ft.Page) -> ft.Control:
    rows = ft.Column(spacing=T.S2, tight=True)
    spinner = ft.ProgressRing(width=14, height=14, visible=False, stroke_width=2, color=T.ACCENT)
    new_msg = ft.Text("", size=11, color=T.TEXT_DIM)

    new_name = ft.TextField(label="Name", width=180, text_size=12,
                            border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
                            content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2))
    new_cron = ft.TextField(label="Cron", hint_text=CRON_HINT, width=240, text_size=12,
                            border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
                            content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2))
    new_prompt = ft.TextField(label="Prompt", expand=True, multiline=True, min_lines=2, max_lines=4,
                              text_size=12,
                              border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
                              content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2))

    def _job_row(j: dict) -> ft.Control:
        name = j.get("name", "?")
        cron = j.get("cron", "?")
        prompt = (j.get("prompt") or "")[:100]
        last = j.get("last_run") or "—"
        last_result = (j.get("last_result") or "").strip()
        active = j.get("active", True)

        async def on_run(_):
            try:
                await api_client.run_schedule_now(j["id"])
            except Exception as ex:
                new_msg.value = f"Run failed: {ex}"
                new_msg.color = T.ERROR
                page.update()
                return
            await refresh()

        async def on_delete(_):
            try:
                await api_client.delete_schedule(j["id"])
            except Exception:
                pass
            await refresh()

        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            bgcolor=T.SURFACE,
            content=ft.Column([
                ft.Row([
                    ft.Container(width=6, height=6, border_radius=3,
                                 bgcolor=T.SUCCESS if active else T.TEXT_FAINT),
                    ft.Text(name, size=13, color=T.TEXT, weight=ft.FontWeight.W_600, width=180),
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                        border=ft.Border.all(1, T.BORDER),
                        border_radius=T.R_SM,
                        content=ft.Text(cron, size=11, color=T.TEXT,
                                        font_family="JetBrains Mono, Consolas, monospace"),
                    ),
                    ft.Container(expand=True),
                    ft.IconButton(icon=ft.Icons.PLAY_ARROW, icon_size=14, icon_color=T.ACCENT,
                                  tooltip="Run now",
                                  on_click=lambda e: page.run_task(on_run, e)),
                    ft.IconButton(icon=ft.Icons.CLOSE, icon_size=14, icon_color=T.TEXT_DIM,
                                  tooltip="Delete",
                                  on_click=lambda e: page.run_task(on_delete, e)),
                ], spacing=T.S3),
                ft.Text(prompt, size=11, color=T.TEXT_DIM),
                ft.Text(f"last run: {last}", size=10, color=T.TEXT_FAINT),
                ft.Text(last_result[:200], size=10, color=T.TEXT_FAINT) if last_result else ft.Container(),
            ], tight=True, spacing=T.S2),
        )

    async def refresh(_e=None) -> None:
        spinner.visible = True
        page.update()
        try:
            data = await api_client.list_schedules()
            jobs = data.get("jobs", [])
            if jobs:
                rows.controls = [_job_row(j) for j in jobs]
            else:
                rows.controls = [ft.Container(
                    padding=ft.Padding.all(T.S5),
                    alignment=ft.Alignment(0, 0),
                    content=ft.Text("No schedules. Create one below.", size=12, color=T.TEXT_FAINT),
                )]
        except Exception as ex:
            rows.controls = [ft.Text(f"Error: {ex}", size=12, color=T.ERROR)]
        finally:
            spinner.visible = False
            page.update()

    async def on_create(_):
        name = (new_name.value or "").strip()
        cron = (new_cron.value or "").strip()
        prompt = (new_prompt.value or "").strip()
        if not (name and cron and prompt):
            new_msg.value = "Name, cron and prompt are all required."
            new_msg.color = T.ERROR
            page.update()
            return
        try:
            await api_client.create_schedule(name, cron, prompt, active=True)
            new_name.value = new_cron.value = new_prompt.value = ""
            new_msg.value = f"✓ Created '{name}'"
            new_msg.color = T.SUCCESS
            page.update()
            await refresh()
        except Exception as ex:
            new_msg.value = f"✗ {ex}"
            new_msg.color = T.ERROR
            page.update()

    page.run_task(refresh)

    return ft.Column([
        ft.Row([
            ft.Text("Schedules", size=18, weight=ft.FontWeight.W_600, color=T.TEXT),
            ft.Container(width=1, height=14, bgcolor=T.BORDER),
            ft.Text("Recurring agent runs · cron-driven", size=12, color=T.TEXT_DIM),
            ft.Container(expand=True),
            ft.IconButton(icon=ft.Icons.REFRESH, icon_size=16, icon_color=T.TEXT_DIM, on_click=refresh),
            spinner,
        ], spacing=T.S3),
        rows,
        ft.Divider(height=1, color=T.BORDER),
        ft.Text("New schedule", size=12, color=T.TEXT_DIM, weight=ft.FontWeight.W_600),
        ft.Container(
            padding=ft.Padding.all(T.S4),
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            bgcolor=T.SURFACE,
            content=ft.Column([
                ft.Row([new_name, new_cron], spacing=T.S2),
                new_prompt,
                ft.Row([
                    ft.ElevatedButton(
                        "Create", icon=ft.Icons.ADD, on_click=on_create,
                        style=ft.ButtonStyle(
                            bgcolor=T.ACCENT, color="#FFFFFF",
                            shape=ft.RoundedRectangleBorder(radius=T.R),
                            padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
                        ),
                    ),
                    new_msg,
                ], spacing=T.S3),
            ], spacing=T.S3, tight=True),
        ),
    ], spacing=T.S4, expand=True, scroll=ft.ScrollMode.AUTO)
