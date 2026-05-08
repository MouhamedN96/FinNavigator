"""Alerts page — list active price alerts, create new ones, route to channels."""

from __future__ import annotations

import flet as ft

from ui import api_client
from ui.theme import T


def build(page: ft.Page) -> ft.Control:
    rows = ft.Column(spacing=T.S2, tight=True)
    spinner = ft.ProgressRing(width=14, height=14, visible=False, stroke_width=2, color=T.ACCENT)
    new_msg = ft.Text("", size=11, color=T.TEXT_DIM)

    new_ticker = ft.TextField(
        label="Ticker", width=120, text_size=12, capitalization=ft.TextCapitalization.CHARACTERS,
        border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
    )
    new_condition = ft.Dropdown(
        label="When price",
        value="above", width=140, text_size=12,
        bgcolor=T.SURFACE, border_color=T.BORDER, focused_border_color=T.ACCENT, color=T.TEXT,
        options=[ft.dropdown.Option("above", "rises above"),
                 ft.dropdown.Option("below", "falls below"),
                 ft.dropdown.Option("change_pct", "changes by %")],
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
    )
    new_threshold = ft.TextField(
        label="Threshold", width=120, text_size=12,
        border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
    )
    new_channel = ft.Dropdown(
        label="Send via",
        value="voiceflow", width=140, text_size=12,
        bgcolor=T.SURFACE, border_color=T.BORDER, focused_border_color=T.ACCENT, color=T.TEXT,
        options=[ft.dropdown.Option("voiceflow"),
                 ft.dropdown.Option("slack", "slack (soon)"),
                 ft.dropdown.Option("email", "email (soon)")],
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
    )

    async def refresh(_e=None) -> None:
        spinner.visible = True
        page.update()
        try:
            data = await api_client.list_alerts()
            alerts = data.get("alerts", [])
            if not alerts:
                rows.controls = [ft.Container(
                    padding=ft.Padding.all(T.S5),
                    alignment=ft.Alignment(0, 0),
                    content=ft.Text("No active alerts. Add one below to start.", size=12, color=T.TEXT_FAINT),
                )]
            else:
                rows.controls = [_alert_row(a) for a in alerts]
        except Exception as ex:
            rows.controls = [ft.Text(f"Error: {ex}", size=12, color=T.ERROR)]
        finally:
            spinner.visible = False
            page.update()

    def _alert_row(a: dict) -> ft.Control:
        cond_text = {
            "above": f"rises above {a['threshold']}",
            "below": f"falls below {a['threshold']}",
            "change_pct": f"changes by {a['threshold']}%",
        }.get(a["condition"], a["condition"])

        async def on_delete(_):
            await api_client.delete_alert(a["id"])
            await refresh()

        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            bgcolor=T.SURFACE,
            content=ft.Row([
                ft.Container(width=6, height=6, border_radius=3, bgcolor=T.ACCENT),
                ft.Text(a["ticker"], size=13, color=T.TEXT, weight=ft.FontWeight.W_600, width=70),
                ft.Text(cond_text, size=12, color=T.TEXT_DIM),
                ft.Container(expand=True),
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    border=ft.Border.all(1, T.BORDER),
                    border_radius=T.R_SM,
                    content=ft.Text(a["channel"], size=10, color=T.TEXT_DIM),
                ),
                ft.IconButton(icon=ft.Icons.CLOSE, icon_size=14, icon_color=T.TEXT_DIM,
                              tooltip="Delete alert", on_click=on_delete),
            ], spacing=T.S3),
        )

    async def on_create(_):
        ticker = (new_ticker.value or "").strip().upper()
        thresh = new_threshold.value or ""
        if not ticker or not thresh:
            new_msg.value = "Ticker and threshold required."
            new_msg.color = T.ERROR
            page.update()
            return
        try:
            await api_client.create_alert(
                ticker=ticker,
                condition=new_condition.value,
                threshold=float(thresh),
                channel=new_channel.value,
            )
            new_ticker.value = new_threshold.value = ""
            new_msg.value = f"✓ Alert added for {ticker}"
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
            ft.Text("Alerts", size=18, weight=ft.FontWeight.W_600, color=T.TEXT),
            ft.Container(width=1, height=14, bgcolor=T.BORDER),
            ft.Text("Price-driven notifications", size=12, color=T.TEXT_DIM),
            ft.Container(expand=True),
            ft.IconButton(icon=ft.Icons.REFRESH, icon_size=16, icon_color=T.TEXT_DIM, on_click=refresh),
            spinner,
        ], spacing=T.S3),
        rows,
        ft.Divider(height=1, color=T.BORDER),
        ft.Text("New alert", size=12, color=T.TEXT_DIM, weight=ft.FontWeight.W_600),
        ft.Container(
            padding=ft.Padding.all(T.S4),
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            bgcolor=T.SURFACE,
            content=ft.Column([
                ft.Row([new_ticker, new_condition, new_threshold, new_channel], spacing=T.S2, wrap=True),
                ft.Row([
                    ft.ElevatedButton(
                        "Create alert", icon=ft.Icons.ADD,
                        on_click=on_create,
                        style=ft.ButtonStyle(bgcolor=T.ACCENT, color="#FFFFFF",
                                             shape=ft.RoundedRectangleBorder(radius=T.R),
                                             padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3)),
                    ),
                    new_msg,
                ], spacing=T.S3),
            ], spacing=T.S3, tight=True),
        ),
    ], spacing=T.S4, expand=True, scroll=ft.ScrollMode.AUTO)
