"""Portfolio page — editable positions, click ticker → Research, sector pie + value bars."""

from __future__ import annotations

from typing import Any, Dict, List

import flet as ft
import flet_charts as fc

from ui import api_client
from ui.theme import T

DEFAULT_POSITIONS: List[Dict[str, Any]] = [
    {"ticker": "NVDA", "shares": 50, "avg_cost": 450.0, "current_price": 905.0},
    {"ticker": "AAPL", "shares": 100, "avg_cost": 150.0, "current_price": 178.0},
    {"ticker": "MSFT", "shares": 30, "avg_cost": 300.0, "current_price": 415.0},
    {"ticker": "TSLA", "shares": 25, "avg_cost": 200.0, "current_price": 175.0},
]

# crude sector mapping — agent-driven version comes later
SECTOR_MAP = {
    "NVDA": "Tech", "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech",
    "TSLA": "Consumer", "AMZN": "Consumer", "DIS": "Consumer", "NKE": "Consumer",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "XOM": "Energy", "CVX": "Energy",
}
SECTOR_COLORS = {
    "Tech":        "#5E6AD2",
    "Consumer":    "#E2A73D",
    "Financials":  "#4CB782",
    "Healthcare":  "#EB5757",
    "Energy":      "#9B7BD2",
    "Other":       "#8A8F98",
}


def build(page: ft.Page) -> ft.Control:
    positions: List[Dict[str, Any]] = [dict(p) for p in DEFAULT_POSITIONS]

    table_body = ft.Column(spacing=0, tight=True)
    summary_value = ft.Text("$0", size=22, weight=ft.FontWeight.W_600, color=T.TEXT)
    summary_pl = ft.Text("$0", size=12, color=T.TEXT_DIM)
    pie_chart_holder = ft.Container(content=ft.Container(), height=180, width=180)
    pie_legend = ft.Column(tight=True, spacing=T.S2)
    bar_chart_holder = ft.Container(content=ft.Container(), expand=True, height=180)
    spinner = ft.ProgressRing(width=14, height=14, visible=False, stroke_width=2, color=T.ACCENT)
    result_md = ft.Markdown("Edit positions or run an analysis. Risk metrics show up here.", selectable=True)
    analysis_dd = ft.Dropdown(
        value="overview", width=160, text_size=12,
        bgcolor=T.SURFACE, border_color=T.BORDER, focused_border_color=T.ACCENT, color=T.TEXT,
        options=[ft.dropdown.Option(v) for v in ("overview", "risk", "sector", "rebalance")],
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S3),
    )

    def _open_research(ticker: str) -> None:
        # page.session changed shape in Flet 0.84 — store on page directly instead.
        page.research_prefill_ticker = ticker  # type: ignore[attr-defined]
        if hasattr(page, "navigate"):
            page.navigate("Research")  # type: ignore[attr-defined]

    def _editable(value: str | float, idx: int, key: str, width: int = 90) -> ft.TextField:
        f = ft.TextField(
            value=str(value),
            width=width,
            text_size=12,
            border=ft.InputBorder.NONE,
            bgcolor=T.SURFACE,
            color=T.TEXT,
            text_align=ft.TextAlign.RIGHT,
            content_padding=ft.Padding.symmetric(horizontal=T.S2, vertical=T.S2),
        )

        def on_change(_):
            try:
                positions[idx][key] = float(f.value or 0)
            except ValueError:
                return
            _refresh_totals()
            _refresh_charts()
            page.update()

        f.on_change = on_change
        return f

    def _row(idx: int, p: Dict[str, Any]) -> ft.Control:
        ticker_btn = ft.TextButton(
            content=ft.Text(p["ticker"], size=12, weight=ft.FontWeight.W_600, color=T.TEXT),
            on_click=lambda _, t=p["ticker"]: _open_research(t),
            style=ft.ButtonStyle(
                color=T.TEXT,
                padding=ft.Padding.symmetric(horizontal=T.S2, vertical=T.S1),
                shape=ft.RoundedRectangleBorder(radius=T.R_SM),
            ),
            tooltip="Open in Research",
        )
        delete_btn = ft.IconButton(
            icon=ft.Icons.CLOSE, icon_size=12, icon_color=T.TEXT_FAINT,
            tooltip="Remove",
            on_click=lambda _, i=idx: _remove(i),
        )
        # value & P/L are read-only
        value = p["shares"] * p["current_price"]
        pl = p["shares"] * (p["current_price"] - p["avg_cost"])
        pl_color = T.SUCCESS if pl >= 0 else T.ERROR
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
            border=ft.Border.only(bottom=ft.BorderSide(1, T.BORDER_FAINT)),
            content=ft.Row([
                ft.Container(content=ticker_btn, width=80),
                _editable(p["shares"], idx, "shares", width=80),
                _editable(p["avg_cost"], idx, "avg_cost", width=100),
                _editable(p["current_price"], idx, "current_price", width=100),
                ft.Text(f"{value:,.0f}", size=12, color=T.TEXT, width=100, text_align=ft.TextAlign.RIGHT,
                        font_family="JetBrains Mono, Consolas, monospace"),
                ft.Text(f"{pl:+,.0f}", size=12, color=pl_color, width=100, text_align=ft.TextAlign.RIGHT,
                        font_family="JetBrains Mono, Consolas, monospace"),
                ft.Container(expand=True),
                delete_btn,
            ], spacing=T.S2),
        )

    def _remove(idx: int) -> None:
        if 0 <= idx < len(positions):
            positions.pop(idx)
            _refresh_table()
            _refresh_totals()
            _refresh_charts()
            page.update()

    def _refresh_table() -> None:
        table_body.controls = [_row(i, p) for i, p in enumerate(positions)]

    def _refresh_totals() -> None:
        total_value = sum(p["shares"] * p["current_price"] for p in positions)
        total_pl = sum(p["shares"] * (p["current_price"] - p["avg_cost"]) for p in positions)
        summary_value.value = f"${total_value:,.0f}"
        summary_pl.value = f"{total_pl:+,.0f} P/L  ·  {len(positions)} positions"
        summary_pl.color = T.SUCCESS if total_pl >= 0 else T.ERROR

    def _refresh_charts() -> None:
        # Sector pie
        sector_totals: Dict[str, float] = {}
        for p in positions:
            sector = SECTOR_MAP.get(p["ticker"].upper(), "Other")
            sector_totals[sector] = sector_totals.get(sector, 0) + p["shares"] * p["current_price"]
        total = sum(sector_totals.values()) or 1
        sections = [
            fc.PieChartSection(
                value=v,
                color=SECTOR_COLORS.get(s, T.TEXT_FAINT),
                radius=70,
                title="",
            )
            for s, v in sector_totals.items()
        ]
        pie_chart_holder.content = fc.PieChart(
            sections=sections,
            sections_space=2,
            center_space_radius=32,
            expand=True,
        )
        pie_legend.controls = [
            ft.Row([
                ft.Container(width=10, height=10, border_radius=2, bgcolor=SECTOR_COLORS.get(s, T.TEXT_FAINT)),
                ft.Text(s, size=11, color=T.TEXT, weight=ft.FontWeight.W_500, width=90),
                ft.Text(f"{(v/total*100):.0f}%", size=11, color=T.TEXT_DIM,
                        font_family="JetBrains Mono, Consolas, monospace"),
            ], spacing=T.S2, tight=True)
            for s, v in sorted(sector_totals.items(), key=lambda kv: -kv[1])
        ]

        # Value bars per position
        if not positions:
            bar_chart_holder.content = ft.Container()
            return
        max_val = max(p["shares"] * p["current_price"] for p in positions) or 1
        groups = []
        for i, p in enumerate(positions):
            value = p["shares"] * p["current_price"]
            pl = p["shares"] * (p["current_price"] - p["avg_cost"])
            color = T.SUCCESS if pl >= 0 else T.ERROR
            groups.append(fc.BarChartGroup(
                x=i,
                rods=[fc.BarChartRod(
                    from_y=0,
                    to_y=value,
                    width=18,
                    color=color,
                    border_radius=2,
                    tooltip=f"{p['ticker']}: ${value:,.0f}",
                )],
            ))
        bar_chart_holder.content = fc.BarChart(
            groups=groups,
            border=ft.Border.all(1, T.BORDER_FAINT),
            left_axis=fc.ChartAxis(label_size=30, title=ft.Text("USD", size=10, color=T.TEXT_FAINT)),
            bottom_axis=fc.ChartAxis(
                label_size=30,
                labels=[fc.ChartAxisLabel(value=i, label=ft.Text(p["ticker"], size=10, color=T.TEXT_DIM)) for i, p in enumerate(positions)],
            ),
            max_y=max_val * 1.15,
            interactive=True,
            expand=True,
        )

    new_ticker = ft.TextField(label="Ticker", width=120, text_size=12,
                              capitalization=ft.TextCapitalization.CHARACTERS,
                              border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
                              content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2))
    new_shares = ft.TextField(label="Shares", width=100, text_size=12,
                              border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
                              content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2))
    new_cost = ft.TextField(label="Avg cost", width=110, text_size=12,
                            border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
                            content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2))
    new_price = ft.TextField(label="Current", width=110, text_size=12,
                             border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
                             content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2))

    def on_add(_):
        try:
            positions.append({
                "ticker": (new_ticker.value or "").upper(),
                "shares": float(new_shares.value or 0),
                "avg_cost": float(new_cost.value or 0),
                "current_price": float(new_price.value or 0),
            })
            new_ticker.value = new_shares.value = new_cost.value = new_price.value = ""
            _refresh_table()
            _refresh_totals()
            _refresh_charts()
            page.update()
        except Exception as ex:
            result_md.value = f"**Couldn't add row:** {ex}"
            page.update()

    async def on_analyze(_e=None) -> None:
        spinner.visible = True
        page.update()
        try:
            data = await api_client.portfolio_analyze(positions, analysis_dd.value)
            risk = data.get("risk") or {}
            sector = data.get("sector") or {}
            lines = []
            if risk:
                lines.append(f"**VaR (95%, 1d):** ${risk.get('var_absolute', 0):,.2f} · {risk.get('var_percentage', 0):.2f}%")
                lines.append(f"_{risk.get('interpretation','')}_")
            if sector:
                lines.append("")
                lines.append(f"**Diversification score:** {sector.get('diversification_score', 0)} / 100")
                exp = sector.get("sector_exposure", {})
                if exp:
                    lines.append("")
                    for s, info in sorted(exp.items(), key=lambda kv: -kv[1].get("allocation", 0)):
                        lines.append(f"- **{s}** — {info.get('allocation', 0):.1f}%  (${info.get('value', 0):,.0f})")
            result_md.value = "\n".join(lines) or "_No analysis returned._"
        except Exception as ex:
            result_md.value = f"**Error:** {ex}"
        finally:
            spinner.visible = False
            page.update()

    # Header row
    head = ft.Container(
        padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
        border=ft.Border.only(bottom=ft.BorderSide(1, T.BORDER)),
        bgcolor=T.SURFACE_2,
        content=ft.Row([
            ft.Text("TICKER",  size=10, color=T.TEXT_DIM, weight=ft.FontWeight.W_600, width=80),
            ft.Text("SHARES",  size=10, color=T.TEXT_DIM, weight=ft.FontWeight.W_600, width=80, text_align=ft.TextAlign.RIGHT),
            ft.Text("AVG COST",size=10, color=T.TEXT_DIM, weight=ft.FontWeight.W_600, width=100, text_align=ft.TextAlign.RIGHT),
            ft.Text("CURRENT", size=10, color=T.TEXT_DIM, weight=ft.FontWeight.W_600, width=100, text_align=ft.TextAlign.RIGHT),
            ft.Text("VALUE",   size=10, color=T.TEXT_DIM, weight=ft.FontWeight.W_600, width=100, text_align=ft.TextAlign.RIGHT),
            ft.Text("P/L",     size=10, color=T.TEXT_DIM, weight=ft.FontWeight.W_600, width=100, text_align=ft.TextAlign.RIGHT),
            ft.Container(expand=True),
        ], spacing=T.S2),
    )

    _refresh_table()
    _refresh_totals()
    _refresh_charts()

    return ft.Column([
        ft.Row([
            ft.Text("Portfolio", size=18, weight=ft.FontWeight.W_600, color=T.TEXT),
            ft.Container(width=1, height=14, bgcolor=T.BORDER),
            ft.Column([summary_value, summary_pl], tight=True, spacing=2),
        ], spacing=T.S3),
        ft.Container(
            bgcolor=T.SURFACE,
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            content=ft.Column([head, table_body], tight=True, spacing=0),
        ),
        ft.Container(
            bgcolor=T.SURFACE,
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
            content=ft.Row([
                new_ticker, new_shares, new_cost, new_price,
                ft.IconButton(
                    icon=ft.Icons.ADD, icon_color=T.TEXT, bgcolor=T.SURFACE_2,
                    tooltip="Add position", on_click=on_add,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=T.R)),
                ),
            ], spacing=T.S2),
        ),
        # Charts row
        ft.Row([
            ft.Container(
                bgcolor=T.SURFACE, border=ft.Border.all(1, T.BORDER), border_radius=T.R,
                padding=ft.Padding.all(T.S4), width=380,
                content=ft.Column([
                    ft.Text("Sector exposure", size=11, color=T.TEXT_DIM, weight=ft.FontWeight.W_600),
                    ft.Row([pie_chart_holder, pie_legend], spacing=T.S4, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ], spacing=T.S3, tight=True),
            ),
            ft.Container(
                bgcolor=T.SURFACE, border=ft.Border.all(1, T.BORDER), border_radius=T.R,
                padding=ft.Padding.all(T.S4), expand=True,
                content=ft.Column([
                    ft.Text("Position value", size=11, color=T.TEXT_DIM, weight=ft.FontWeight.W_600),
                    bar_chart_holder,
                ], spacing=T.S3, tight=True),
            ),
        ], spacing=T.S3),
        ft.Row([
            analysis_dd,
            ft.ElevatedButton(
                "Run analysis", icon=ft.Icons.PLAY_ARROW, on_click=on_analyze,
                style=ft.ButtonStyle(
                    bgcolor=T.ACCENT, color="#FFFFFF",
                    shape=ft.RoundedRectangleBorder(radius=T.R),
                    padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
                ),
            ),
            spinner,
        ], spacing=T.S3),
        ft.Container(
            bgcolor=T.SURFACE, border=ft.Border.all(1, T.BORDER), border_radius=T.R,
            padding=ft.Padding.all(T.S4), expand=True,
            content=result_md,
        ),
    ], spacing=T.S4, expand=True, scroll=ft.ScrollMode.AUTO)
