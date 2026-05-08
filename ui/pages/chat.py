"""Chat page — streaming SSE consumer with history replay, tool toggles, and file upload."""

from __future__ import annotations

import os
import re

import flet as ft

from ui import api_client
from ui.theme import T


SUGGESTED_PROMPTS = [
    "Compare AAPL and MSFT risk factors in their latest 10-K",
    "What does Form 4 disclose?",
    "If a company's revenue grew 12% YoY, what does that imply about operating leverage?",
    "Classify: Q3 revenue beat consensus by 8% and full-year guidance was raised",
]


def build(page: ft.Page) -> ft.Control:
    transcript = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=T.S3)
    spinner = ft.ProgressRing(width=14, height=14, visible=False, stroke_width=2, color=T.ACCENT)
    streaming_dot = ft.Container(width=6, height=6, border_radius=3, bgcolor=T.ACCENT, visible=False)
    text_input = ft.TextField(
        hint_text="Ask anything — financial reasoning, SEC filings, portfolio…",
        expand=True,
        multiline=False,
        border_color=T.BORDER,
        focused_border_color=T.ACCENT,
        bgcolor=T.SURFACE,
        cursor_color=T.ACCENT,
        text_size=13,
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S3),
    )
    send_btn = ft.IconButton(
        icon=ft.Icons.ARROW_UPWARD,
        icon_size=16,
        icon_color=T.TEXT,
        bgcolor=T.SURFACE_2,
        tooltip="Send (↵)",
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=T.R)),
    )
    upload_btn = ft.IconButton(
        icon=ft.Icons.ATTACH_FILE,
        icon_size=16,
        icon_color=T.TEXT_DIM,
        tooltip="Upload a file to the knowledge base",
    )
    tool_pills_row = ft.Row(spacing=T.S2, wrap=True, run_spacing=T.S2)

    # ---------- file upload dialog (FilePicker plugin doesn't work in `python -m ui.app`) ----------
    upload_path = ft.TextField(
        label="File path",
        hint_text=r"C:\path\to\file.pdf  (or .txt / .md / .docx / .csv)",
        text_size=12,
        border_color=T.BORDER, focused_border_color=T.ACCENT, bgcolor=T.SURFACE,
        content_padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
    )
    upload_status = ft.Text("", size=11, color=T.TEXT_DIM)

    async def do_upload(_):
        path = (upload_path.value or "").strip().strip('"')
        if not path or not os.path.isfile(path):
            upload_status.value = f"Not a file: {path or '(empty)'}"
            upload_status.color = T.ERROR
            page.update(); return
        upload_status.value = "Uploading…"
        upload_status.color = T.TEXT_DIM
        page.update()
        try:
            with open(path, "rb") as f:
                data = f.read()
            result = await api_client.upload_file(os.path.basename(path), data)
            if result.get("indexed"):
                upload_status.value = f"✓ Indexed {result['filename']} — {result['chunks']} chunks ({result['size']/1024:.1f} KB)"
                upload_status.color = T.SUCCESS
            else:
                upload_status.value = f"⚠ Saved but not indexed: {result.get('error', 'unknown')}"
                upload_status.color = T.WARNING
        except Exception as ex:
            upload_status.value = f"✗ {ex}"
            upload_status.color = T.ERROR
        page.update()

    def _close_upload(_e=None):
        upload_dialog.open = False
        page.update()

    upload_dialog = ft.AlertDialog(
        modal=True,
        bgcolor=T.SURFACE,
        title=ft.Text("Upload to knowledge base", size=14, weight=ft.FontWeight.W_600, color=T.TEXT),
        content=ft.Container(
            width=480,
            content=ft.Column([
                ft.Text(
                    "File goes to data/uploads/, gets chunked, and indexed into the RAG vectorstore. "
                    "Subsequent agent calls will see it via knowledge_search.",
                    size=11, color=T.TEXT_DIM,
                ),
                upload_path,
                upload_status,
            ], tight=True, spacing=T.S3),
        ),
        actions=[
            ft.TextButton(content=ft.Text("Upload", color=T.ACCENT),
                          on_click=lambda e: page.run_task(do_upload, e)),
            ft.TextButton(content=ft.Text("Close"), on_click=_close_upload),
        ],
    )
    page.overlay.append(upload_dialog)

    def _open_upload(_e=None):
        upload_path.value = ""
        upload_status.value = ""
        upload_dialog.open = True
        page.update()

    upload_btn.on_click = _open_upload

    has_messages = [False]

    # ---------- empty state with suggested prompts ----------
    def empty_state() -> ft.Control:
        async def fire(prompt: str) -> None:
            text_input.value = prompt
            page.update()
            await send(None)

        chips = [
            ft.Container(
                padding=ft.Padding.symmetric(horizontal=T.S3, vertical=T.S2),
                border=ft.Border.all(1, T.BORDER),
                border_radius=T.R_LG,
                bgcolor=T.SURFACE,
                content=ft.Text(p, size=12, color=T.TEXT, no_wrap=False),
                on_click=lambda _, prompt=p: page.run_task(fire, prompt),
                on_hover=lambda e: setattr(e.control, "bgcolor", T.SURFACE_2 if e.data == "true" else T.SURFACE) or e.control.update(),
                width=380,
            )
            for p in SUGGESTED_PROMPTS
        ]
        return ft.Container(
            alignment=ft.Alignment(0, 0),
            expand=True,
            content=ft.Column([
                ft.Text("Start a conversation", size=15, color=T.TEXT_DIM, weight=ft.FontWeight.W_500),
                ft.Text("Pick a prompt or write your own.", size=12, color=T.TEXT_FAINT),
                ft.Container(height=T.S4),
                ft.Row(chips[:2], spacing=T.S2, alignment=ft.MainAxisAlignment.CENTER, wrap=True),
                ft.Row(chips[2:], spacing=T.S2, alignment=ft.MainAxisAlignment.CENTER, wrap=True),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True, spacing=T.S2),
        )

    transcript.controls.append(empty_state())

    def _ensure_messages_visible() -> None:
        if not has_messages[0]:
            transcript.controls.clear()
            has_messages[0] = True

    # ---------- bubble factory (extracted so streaming can mutate Markdown) ----------
    def _make_bubble(role: str, initial_content: str = "") -> tuple[ft.Container, ft.Markdown, ft.Column]:
        is_user = role == "user"
        md = ft.Markdown(
            initial_content,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        )
        col = ft.Column([
            ft.Text(
                "You" if is_user else "Agent",
                size=10,
                weight=ft.FontWeight.W_600,
                color=T.TEXT_FAINT,
            ),
            md,
        ], tight=True, spacing=T.S2)
        container = ft.Container(
            content=col,
            padding=ft.Padding.symmetric(horizontal=T.S4, vertical=T.S3),
            bgcolor=T.SURFACE if not is_user else None,
            border=ft.Border.all(1, T.BORDER if not is_user else T.BORDER_FAINT),
            border_radius=T.R,
            margin=ft.Margin.only(left=80 if is_user else 0, right=0 if is_user else 80),
        )
        return container, md, col

    def _attach_reasoning(content_col: ft.Column, reasoning_steps: list) -> None:
        chips = _reasoning_chips(reasoning_steps or [])
        if not chips:
            return
        content_col.controls.append(ft.Container(
            margin=ft.Margin.only(top=T.S2),
            content=ft.Column([
                ft.Text("Reasoning", size=10, color=T.TEXT_FAINT, weight=ft.FontWeight.W_600),
                ft.Row(chips, wrap=True, spacing=T.S2, run_spacing=T.S2),
            ], tight=True, spacing=T.S2),
        ))

    def add_user_bubble(content: str) -> None:
        _ensure_messages_visible()
        bubble, _, _ = _make_bubble("user", content)
        transcript.controls.append(bubble)

    def add_assistant_bubble(content: str, reasoning_steps: list | None = None) -> None:
        _ensure_messages_visible()
        bubble, _, col = _make_bubble("assistant", content)
        if reasoning_steps:
            _attach_reasoning(col, reasoning_steps)
        transcript.controls.append(bubble)

    # ---------- send (streaming with non-streaming fallback) ----------
    async def send(_e=None) -> None:
        prompt = (text_input.value or "").strip()
        if not prompt:
            return

        add_user_bubble(prompt)
        text_input.value = ""
        spinner.visible = True
        streaming_dot.visible = True
        text_input.disabled = send_btn.disabled = True
        page.update()

        bubble, md, content_col = _make_bubble("assistant", "")
        transcript.controls.append(bubble)
        page.update()

        try:
            errored = False
            async for ev in api_client.agent_stream(prompt):
                t = ev.get("type")
                if t == "token":
                    md.value = (md.value or "") + (ev.get("content") or "")
                    page.update()
                elif t == "done":
                    break
                elif t == "error":
                    md.value = f"**Error:** {ev.get('message', 'unknown')}"
                    page.update()
                    errored = True
                    break

            if not errored and not (md.value or "").strip():
                try:
                    resp = await api_client.agent_run(prompt)
                    md.value = resp.get("content") or "_(empty response)_"
                    _attach_reasoning(content_col, resp.get("reasoning_steps") or [])
                except Exception as ex:
                    md.value = f"**Error:** {ex}"
                page.update()
        except Exception as stream_ex:
            try:
                resp = await api_client.agent_run(prompt)
                md.value = resp.get("content") or f"_(stream failed: {stream_ex})_"
                _attach_reasoning(content_col, resp.get("reasoning_steps") or [])
            except Exception as ex:
                md.value = f"**Error:** {ex}"
        finally:
            spinner.visible = False
            streaming_dot.visible = False
            text_input.disabled = send_btn.disabled = False
            page.update()

    text_input.on_submit = send
    send_btn.on_click = send

    # ---------- history replay ----------
    async def _load_history() -> None:
        try:
            data = await api_client.get_history(limit=100)
        except Exception:
            return
        turns = data.get("turns", [])
        if not turns:
            return
        # Drop the empty state, render past turns
        if not has_messages[0]:
            transcript.controls.clear()
            has_messages[0] = True
        for t in turns:
            role = t.get("role", "assistant")
            content = t.get("content", "")
            reasoning = t.get("reasoning")
            if role == "user":
                add_user_bubble(content)
            else:
                add_assistant_bubble(content, reasoning_steps=reasoning)
        page.update()

    async def _clear_history(_e=None) -> None:
        try:
            await api_client.clear_history()
        except Exception:
            pass
        transcript.controls.clear()
        has_messages[0] = False
        transcript.controls.append(empty_state())
        page.update()

    # ---------- tool pills ----------
    pill_state: dict[str, bool] = {}

    def _pill(name: str, enabled: bool) -> ft.Control:
        bg = T.ACCENT if enabled else T.SURFACE
        fg = "#FFFFFF" if enabled else T.TEXT_DIM
        border = T.ACCENT if enabled else T.BORDER

        async def toggle(_):
            pill_state[name] = not pill_state[name]
            try:
                await api_client.set_enabled_tools([n for n, on in pill_state.items() if on])
            except Exception:
                pill_state[name] = not pill_state[name]  # rollback
            _render_pills()

        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=T.S3, vertical=4),
            border=ft.Border.all(1, border),
            border_radius=T.R_LG,
            bgcolor=bg,
            content=ft.Text(name, size=11, color=fg,
                            weight=ft.FontWeight.W_500,
                            font_family="JetBrains Mono, Consolas, monospace"),
            on_click=lambda e: page.run_task(toggle, e),
            tooltip=("Click to disable" if enabled else "Click to enable"),
        )

    def _render_pills() -> None:
        tool_pills_row.controls = [
            _pill(name, on)
            for name, on in sorted(pill_state.items())
        ]
        page.update()

    async def _load_tool_state() -> None:
        try:
            data = await api_client.get_enabled_tools()
        except Exception:
            return
        all_tools = data.get("all", [])
        enabled = set(data.get("enabled", []))
        pill_state.clear()
        for name in all_tools:
            pill_state[name] = name in enabled
        _render_pills()

    page.run_task(_load_history)
    page.run_task(_load_tool_state)

    # ---------- header + layout ----------
    header = ft.Row([
        ft.Text("Chat", size=18, weight=ft.FontWeight.W_600, color=T.TEXT),
        ft.Container(width=1, height=14, bgcolor=T.BORDER),
        ft.Text("Streaming · persistent · tools toggleable", size=12, color=T.TEXT_DIM),
        ft.Container(expand=True),
        ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_size=16,
            icon_color=T.TEXT_DIM,
            tooltip="Clear chat history",
            on_click=lambda e: page.run_task(_clear_history, e),
        ),
    ], spacing=T.S3)

    return ft.Column([
        header,
        ft.Container(
            content=transcript,
            expand=True,
            padding=ft.Padding.symmetric(horizontal=0, vertical=T.S4),
        ),
        # Tool toggle row
        ft.Container(
            padding=ft.Padding.symmetric(horizontal=T.S2, vertical=T.S2),
            content=ft.Column([
                ft.Text("TOOLS", size=9, color=T.TEXT_FAINT, weight=ft.FontWeight.W_600),
                tool_pills_row,
            ], tight=True, spacing=T.S2),
        ),
        # Input
        ft.Container(
            padding=ft.Padding.all(T.S2),
            border=ft.Border.all(1, T.BORDER),
            border_radius=T.R,
            bgcolor=T.SURFACE,
            content=ft.Row([upload_btn, text_input, streaming_dot, spinner, send_btn], spacing=T.S2),
        ),
    ], expand=True, spacing=T.S4)


# ---- reasoning trace → tool-call chips (used by /api/agent/run fallback only) ----

_TOOL_RE = re.compile(r"(?:Action|Tool|tool|Delegating to|Reasoning:\s*Decomposing|action)[:\s]+([a-zA-Z0-9_\-]+)")
_TOOL_COLORS = {
    "sec":          ("#3A4A8E", "#A6B4F0"),
    "calculator":   ("#3F5D43", "#B6E0BD"),
    "datetime":     ("#3F5D43", "#B6E0BD"),
    "knowledge":    ("#5D4631", "#F2C99A"),
    "vision":       ("#5D3A4F", "#F0A6CC"),
    "send_message": ("#3D5C5C", "#A6E0E0"),
    "send_alert":   ("#5C3F3F", "#F0A6A6"),
    "research":     ("#3D4659", "#C5CFE8"),
    "financial":    ("#3F5D43", "#B6E0BD"),
    "analyst":      ("#3D4659", "#C5CFE8"),
}


def _tool_color(name: str) -> tuple[str, str]:
    n = name.lower()
    for prefix, color in _TOOL_COLORS.items():
        if n.startswith(prefix) or prefix in n:
            return color
    return ("#3A3A40", "#C5C7CC")


def _reasoning_chips(steps: list) -> list[ft.Control]:
    chips: list[ft.Control] = []
    seen = set()
    for s in steps[-12:]:
        text = ""
        if isinstance(s, dict):
            if "raw" in s:
                text = str(s["raw"])
            else:
                text = " · ".join(str(v) for k, v in s.items() if v)
        else:
            text = str(s)
        m = _TOOL_RE.search(text)
        if m:
            tool = m.group(1)
            label = tool
        elif "classified" in text.lower():
            tool = text.split("classified as:")[-1].strip(" .") or "classified"
            label = f"task: {tool}"
        else:
            tool = "step"
            label = text[:48] + ("…" if len(text) > 48 else "")
        key = (tool, label)
        if key in seen:
            continue
        seen.add(key)
        bg, fg = _tool_color(tool)
        chips.append(ft.Container(
            padding=ft.Padding.symmetric(horizontal=8, vertical=3),
            border_radius=T.R_SM,
            bgcolor=bg,
            content=ft.Text(label, size=10, color=fg, weight=ft.FontWeight.W_500),
        ))
    return chips
