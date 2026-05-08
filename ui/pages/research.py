"""Research page — SEC filings deep dive with glossary tooltips on form types."""

from __future__ import annotations

import json

import flet as ft

from ui import api_client
from ui.glossary import lookup as glossary_lookup
from ui.theme import T


def _input(label: str, hint: str = "", width: int | None = None, **kw) -> ft.TextField:
    return ft.TextField(
        label=label, hint_text=hint, width=width, text_size=13,
        label_style=ft.TextStyle(size=11, color=T.TEXT_DIM),
        border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
        cursor_color=T.ACCENT,
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S3),
        **kw,
    )


def _dd(label: str, value: str, opts: list[str], width: int) -> ft.Dropdown:
    return ft.Dropdown(
        label=label, value=value, width=width, text_size=13,
        label_style=ft.TextStyle(size=11, color=T.TEXT_DIM),
        border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
        color=T.TEXT,
        options=[ft.dropdown.Option(v) for v in opts],
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S3),
    )


def build(page: ft.Page) -> ft.Control:
    # Read prefill attribute set by Portfolio's ticker click handler.
    prefill_ticker = getattr(page, "research_prefill_ticker", "") or ""
    if prefill_ticker:
        page.research_prefill_ticker = ""  # type: ignore[attr-defined]  # consume

    ticker_input = _input(
        "Ticker", "NVDA, AAPL, TSLA", width=160,
        capitalization=ft.TextCapitalization.CHARACTERS,
        value=prefill_ticker,
    )
    focus_dd = _dd("Focus", "overview", ["overview", "risks", "financials", "comparison"], width=180)
    depth_dd = _dd("Depth", "standard", ["quick", "standard", "deep"], width=140)
    spinner = ft.ProgressRing(width=14, height=14, visible=False, stroke_width=2, color=T.ACCENT)

    summary_md = ft.Markdown("", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)
    filings_col = ft.Column(spacing=T.S2, tight=True)
    raw_text = ft.Text("", font_family="JetBrains Mono, Consolas, monospace", size=11, color=T.TEXT_DIM, selectable=True)

    def form_chip(form_type: str) -> ft.Control:
        definition = glossary_lookup(form_type) or "No glossary entry. Ask the agent in Chat for details."
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=8, vertical=3),
            border_radius=T.R_SM,
            border=ft.Border.all(1, T.BORDER),
            bgcolor=T.SURFACE_2,
            content=ft.Text(form_type, size=11, color=T.TEXT, weight=ft.FontWeight.W_600,
                            font_family="JetBrains Mono, Consolas, monospace"),
            tooltip=ft.Tooltip(
                message=definition,
                bgcolor=T.SURFACE_2,
                text_style=ft.TextStyle(size=11, color=T.TEXT),
                padding=10,
                border_radius=T.R,
                wait_duration=200,
            ),
        )

    def filing_row(f: dict) -> ft.Control:
        form_type = f.get("formType", f.get("type", "?"))
        filed_at = (f.get("filedAt") or f.get("dateFiled") or "")[:10]
        desc = f.get("description") or f.get("companyName") or f.get("ticker") or ""
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
            border=ft.Border.all(1, T.BORDER_FAINT),
            border_radius=T.R_SM,
            bgcolor=T.SURFACE,
            content=ft.Row([
                form_chip(form_type),
                ft.Text(filed_at, size=11, color=T.TEXT_DIM,
                        font_family="JetBrains Mono, Consolas, monospace", width=90),
                ft.Text(str(desc)[:120], size=12, color=T.TEXT, expand=True),
            ], spacing=T.S3),
        )

    async def go(_e=None) -> None:
        t = (ticker_input.value or "").strip().upper()
        if not t:
            return
        spinner.visible = True
        summary_md.value = ""
        filings_col.controls = []
        raw_text.value = ""
        page.update()
        try:
            data = await api_client.research(t, focus_dd.value, depth_dd.value)
            summary_md.value = data.get("summary") or "_no summary returned_"
            findings = data.get("findings", {}) or {}
            filings_block = findings.get("filings", {})
            if isinstance(filings_block, dict):
                filings = filings_block.get("filings", [])
            elif isinstance(filings_block, list):
                filings = filings_block
            else:
                filings = []
            if filings:
                filings_col.controls = [filing_row(f) for f in filings[:20]]
            else:
                filings_col.controls = [ft.Text("No filings returned.", size=12, color=T.TEXT_FAINT)]
            raw_text.value = json.dumps(findings, indent=2)[:4000]
        except Exception as ex:
            summary_md.value = f"**Error:** {ex}"
        finally:
            spinner.visible = False
            page.update()

    def panel(title: str, body: ft.Control, expand: bool = False) -> ft.Control:
        return ft.Container(
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            bgcolor=T.SURFACE,
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

    # Auto-fire if prefilled
    if prefill_ticker:
        page.run_task(go)

    return ft.Column([
        ft.Row([
            ft.Text("Research", size=18, weight=ft.FontWeight.W_600, color=T.TEXT),
            ft.Container(width=1, height=14, bgcolor=T.BORDER),
            ft.Text("SEC filings · hover form types for definitions", size=12, color=T.TEXT_DIM),
        ], spacing=T.S3),
        ft.Row([
            ticker_input, focus_dd, depth_dd,
            ft.ElevatedButton(
                "Research", icon=ft.Icons.ARROW_FORWARD, on_click=go,
                style=ft.ButtonStyle(
                    bgcolor=T.ACCENT, color="#FFFFFF",
                    shape=ft.RoundedRectangleBorder(radius=T.R),
                    padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
                ),
            ),
            spinner,
        ], spacing=T.S3),
        panel("Summary", summary_md),
        panel("Recent filings", filings_col),
        ft.ExpansionTile(
            title=ft.Text("Raw findings", size=12, color=T.TEXT_DIM),
            controls=[
                ft.Container(
                    padding=ft.Padding.all(T.S4),
                    border=ft.Border.all(1, T.BORDER_FAINT),
                    border_radius=T.R_SM,
                    bgcolor=T.SURFACE,
                    content=raw_text,
                ),
            ],
        ),
    ], spacing=T.S4, expand=True, scroll=ft.ScrollMode.AUTO)
