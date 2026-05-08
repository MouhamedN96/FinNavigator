"""Flet entry — Linear-style shell with sidebar, backend switcher,
keyboard shortcuts (C/R/P/M/A + Cmd+K palette), and channels drawer."""

from __future__ import annotations

import flet as ft

from ui import api_client
from ui.pages import alerts as alerts_page
from ui.pages import chat, monitor, portfolio, research
from ui.pages import schedules as schedules_page
from ui.theme import T, linear_theme

PAGES = [
    ("Chat",      "C", chat.build),
    ("Research",  "R", research.build),
    ("Portfolio", "P", portfolio.build),
    ("Monitor",   "M", monitor.build),
    ("Alerts",    "A", alerts_page.build),
    ("Schedules", "S", schedules_page.build),
]


def main(page: ft.Page) -> None:
    page.title = "FinNavigator"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = linear_theme()
    page.bgcolor = T.BG
    page.padding = 0
    page.window.width = 1280
    page.window.height = 820

    body = ft.Container(
        content=PAGES[0][2](page),
        expand=True,
        padding=ft.Padding.all(T.S6),
        bgcolor=T.BG,
    )

    status_dot = ft.Container(width=6, height=6, border_radius=3, bgcolor=T.TEXT_FAINT)
    backend_label = ft.Text("…", size=11, color=T.TEXT_DIM, weight=ft.FontWeight.W_500)
    backend_dropdown = ft.Dropdown(
        value=None,
        options=[],
        width=140,
        height=32,
        text_size=12,
        content_padding=ft.Padding.symmetric(horizontal=8, vertical=0),
        bgcolor=T.SURFACE,
        border_color=T.BORDER,
        focused_border_color=T.ACCENT,
        color=T.TEXT,
    )

    active_idx = [0]
    nav_items: list[ft.Container] = []

    def render_body(idx: int) -> None:
        active_idx[0] = idx
        body.content = PAGES[idx][2](page)
        for i, item in enumerate(nav_items):
            item.bgcolor = T.SURFACE_2 if i == idx else None
            item.update()
        page.update()

    # Pages can call page.navigate("Research") to jump cross-tab
    def navigate(label: str) -> None:
        for i, (name, _, _) in enumerate(PAGES):
            if name.lower() == label.lower():
                render_body(i)
                return
    page.navigate = navigate  # type: ignore[attr-defined]

    def make_nav_item(idx: int, label: str, key: str) -> ft.Container:
        def on_click(_):
            render_body(idx)

        def on_hover(e: ft.HoverEvent):
            if active_idx[0] != idx:
                e.control.bgcolor = T.SURFACE if e.data == "true" else None
                e.control.update()

        return ft.Container(
            content=ft.Row([
                ft.Text(label, size=13, color=T.TEXT, weight=ft.FontWeight.W_500),
                ft.Container(expand=True),
                ft.Container(
                    content=ft.Text(key, size=10, color=T.TEXT_FAINT, weight=ft.FontWeight.W_600),
                    padding=ft.Padding.symmetric(horizontal=5, vertical=1),
                    border=ft.Border.all(1, T.BORDER),
                    border_radius=T.R_SM,
                ),
            ], tight=True),
            padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
            border_radius=T.R,
            on_click=on_click,
            on_hover=on_hover,
            bgcolor=T.SURFACE_2 if idx == 0 else None,
        )

    nav_items = [make_nav_item(i, label, key) for i, (label, key, _) in enumerate(PAGES)]

    # ---------------- Backend selector / config refresh ----------------
    async def refresh_config() -> None:
        try:
            cfg = await api_client.get_config()
        except Exception:
            backend_label.value = "offline"
            status_dot.bgcolor = T.ERROR
            page.update()
            return
        backend_label.value = cfg["current"]
        status_dot.bgcolor = T.SUCCESS
        backend_dropdown.value = cfg["current"]
        backend_dropdown.options = [
            ft.dropdown.Option(b, b.upper() if b == "nvidia" else b.title())
            for b in cfg["supported"]
        ]
        page.update()

    async def on_backend_change(e: ft.ControlEvent) -> None:
        new = e.control.value
        backend_label.value = "switching…"
        status_dot.bgcolor = T.WARNING
        page.update()
        try:
            await api_client.set_backend(new)
            backend_label.value = new
            status_dot.bgcolor = T.SUCCESS
            page.snack_bar = ft.SnackBar(content=ft.Text(f"Backend → {new}"), bgcolor=T.SURFACE_2)
            page.snack_bar.open = True
        except Exception as ex:
            backend_label.value = "error"
            status_dot.bgcolor = T.ERROR
            page.snack_bar = ft.SnackBar(content=ft.Text(f"Switch failed: {ex}"), bgcolor=T.ERROR)
            page.snack_bar.open = True
        finally:
            page.update()

    backend_dropdown.on_change = on_backend_change

    # ---------------- Cmd/Ctrl + K command palette ----------------
    palette_field = ft.TextField(
        hint_text="Jump to … (chat, research, portfolio, monitor, alerts, channels)",
        autofocus=True,
        border_color=T.BORDER,
        focused_border_color=T.ACCENT,
        bgcolor=T.SURFACE,
        text_size=13,
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S3),
    )
    palette_results = ft.Column(spacing=2, tight=True)

    def palette_close() -> None:
        palette_dialog.open = False
        page.update()

    def palette_action(label: str) -> None:
        palette_close()
        for i, (name, _, _) in enumerate(PAGES):
            if name.lower() == label.lower():
                render_body(i)
                return
        if label.lower() == "channels":
            channels_open(None)

    def palette_render(query: str) -> None:
        q = query.strip().lower()
        all_actions = [name for name, _, _ in PAGES] + ["Channels"]
        hits = [a for a in all_actions if q in a.lower()] if q else all_actions
        palette_results.controls = [
            ft.Container(
                padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
                border_radius=T.R,
                content=ft.Row([
                    ft.Text(label, size=13, color=T.TEXT),
                    ft.Container(expand=True),
                    ft.Text("↵", size=11, color=T.TEXT_FAINT),
                ]),
                on_click=lambda _, lbl=label: palette_action(lbl),
                on_hover=lambda e: setattr(e.control, "bgcolor", T.SURFACE_2 if e.data == "true" else None) or e.control.update(),
            )
            for label in hits
        ]
        palette_results.update()

    palette_field.on_change = lambda e: palette_render(e.control.value or "")
    palette_field.on_submit = lambda e: palette_action((palette_field.value or "").strip())

    palette_dialog = ft.AlertDialog(
        modal=True,
        bgcolor=T.SURFACE,
        content=ft.Container(
            width=520,
            content=ft.Column([palette_field, ft.Divider(height=1, color=T.BORDER), palette_results], tight=True, spacing=T.S2),
        ),
    )

    def palette_open() -> None:
        palette_field.value = ""
        palette_render("")
        palette_dialog.open = True
        page.update()

    # ---------------- Channels settings drawer ----------------
    channels_list = ft.Column(tight=True, spacing=T.S2)
    test_user_field = ft.TextField(
        label="User ID", value="finnav_test_user", text_size=12, width=200,
        border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
    )
    test_msg_field = ft.TextField(
        label="Test message", value="FinNavigator test ping.", text_size=12, expand=True,
        border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
    )
    test_status = ft.Text("", size=11, color=T.TEXT_DIM)

    async def refresh_channels() -> None:
        try:
            data = await api_client.list_channels()
        except Exception as ex:
            channels_list.controls = [ft.Text(f"Error: {ex}", size=12, color=T.ERROR)]
            page.update()
            return
        rows = []
        for ch in data.get("channels", []):
            ok = ch["configured"]
            rows.append(ft.Container(
                padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
                border_radius=T.R,
                bgcolor=T.SURFACE,
                content=ft.Row([
                    ft.Container(width=6, height=6, border_radius=3, bgcolor=T.SUCCESS if ok else T.TEXT_FAINT),
                    ft.Text(ch["name"], size=13, color=T.TEXT, weight=ft.FontWeight.W_500),
                    ft.Container(expand=True),
                    ft.Text("configured" if ok else "missing key", size=11, color=T.SUCCESS if ok else T.TEXT_FAINT),
                ]),
            ))
        channels_list.controls = rows
        page.update()

    async def on_test_send(_):
        test_status.value = "sending…"
        test_status.color = T.TEXT_DIM
        page.update()
        try:
            data = await api_client.test_channel(test_user_field.value or "finnav_test_user", test_msg_field.value or "ping")
            if data.get("ok"):
                test_status.value = f"✓ {data.get('result','sent')}"
                test_status.color = T.SUCCESS
            else:
                test_status.value = f"✗ {data.get('error','failed')}"
                test_status.color = T.ERROR
        except Exception as ex:
            test_status.value = f"✗ {ex}"
            test_status.color = T.ERROR
        finally:
            page.update()

    channels_drawer = ft.AlertDialog(
        modal=True,
        bgcolor=T.SURFACE,
        title=ft.Text("Channels", size=14, weight=ft.FontWeight.W_600, color=T.TEXT),
        content=ft.Container(
            width=520,
            content=ft.Column(
                [
                    ft.Text("Outbound messaging endpoints. Voiceflow is the only one currently wired; others land later.",
                            size=11, color=T.TEXT_DIM),
                    channels_list,
                    ft.Divider(height=1, color=T.BORDER),
                    ft.Text("Test send", size=12, weight=ft.FontWeight.W_600, color=T.TEXT),
                    ft.Row([test_user_field, test_msg_field], spacing=T.S2),
                    ft.Row([
                        ft.ElevatedButton("Send test", icon=ft.Icons.SEND, on_click=on_test_send,
                                          style=ft.ButtonStyle(bgcolor=T.ACCENT, color="#FFFFFF",
                                                               shape=ft.RoundedRectangleBorder(radius=T.R))),
                        test_status,
                    ], spacing=T.S3),
                ],
                tight=True, spacing=T.S3,
            ),
        ),
        actions=[ft.TextButton(content=ft.Text("Close"), on_click=lambda _: _close_channels())],
    )

    def _close_channels() -> None:
        channels_drawer.open = False
        page.update()

    def channels_open(_e):
        channels_drawer.open = True
        page.update()
        page.run_task(refresh_channels)

    # ---------------- Keyboard ----------------
    def on_key(e: ft.KeyboardEvent) -> None:
        # Cmd/Ctrl + K → palette
        if (e.meta or e.ctrl) and e.key.lower() == "k":
            palette_open()
            return
        # Plain letters → nav (only when no modifier and no input focus)
        if e.shift or e.ctrl or e.alt or e.meta:
            return
        keymap = {"c": 0, "r": 1, "p": 2, "m": 3, "a": 4, "s": 5}
        target = keymap.get(e.key.lower())
        if target is not None:
            render_body(target)

    page.on_keyboard_event = on_key

    # ---------------- Sidebar layout ----------------
    sidebar = ft.Container(
        width=220,
        bgcolor=T.SIDEBAR,
        border=ft.Border.only(right=ft.BorderSide(1, T.BORDER)),
        padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S4),
        content=ft.Column([
            ft.Container(
                padding=ft.Padding.only(left=T.S2, bottom=T.S5),
                content=ft.Row([
                    ft.Container(width=14, height=14, bgcolor=T.ACCENT, border_radius=3),
                    ft.Text("FinNavigator", size=13, weight=ft.FontWeight.W_600, color=T.TEXT),
                ], spacing=T.S2),
            ),
            ft.Column(nav_items, spacing=2, tight=True),
            ft.Container(
                padding=ft.Padding.only(top=T.S3, left=T.S2, right=T.S2),
                content=ft.Row([
                    ft.Text("⌘K", size=10, color=T.TEXT_FAINT, weight=ft.FontWeight.W_600),
                    ft.Text("palette", size=10, color=T.TEXT_FAINT),
                    ft.Container(expand=True),
                    ft.IconButton(
                        icon=ft.Icons.SETTINGS_OUTLINED, icon_size=14, icon_color=T.TEXT_DIM,
                        tooltip="Channels", on_click=channels_open,
                    ),
                ], tight=True),
            ),
            ft.Container(expand=True),
            ft.Container(
                border=ft.Border.only(top=ft.BorderSide(1, T.BORDER)),
                padding=ft.Padding.only(top=T.S3),
                content=ft.Column([
                    ft.Row([
                        status_dot,
                        ft.Text("Backend", size=11, color=T.TEXT_DIM, weight=ft.FontWeight.W_500),
                        ft.Container(expand=True),
                        backend_label,
                    ], spacing=T.S2, tight=True),
                    backend_dropdown,
                ], spacing=T.S2, tight=True),
            ),
        ], tight=True, expand=True),
    )

    # Mount dialogs in the overlay so .open=True actually shows them
    page.overlay.append(palette_dialog)
    page.overlay.append(channels_drawer)

    page.add(ft.Row([sidebar, body], expand=True, spacing=0))
    page.run_task(refresh_config)


if __name__ == "__main__":
    ft.run(main)
