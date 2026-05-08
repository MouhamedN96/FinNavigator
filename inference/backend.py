"""LLM backend adapter — single entry point shared by FastAPI server, Flet desktop, tests.

Extracted from `app.py:get_llm_client()` so callers don't have to depend on Streamlit.
The function raises `BackendError` instead of writing to a UI; let callers translate.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

Backend = Literal["ollama", "openai", "anthropic", "nvidia", "llama", "webgpu"]
SUPPORTED_BACKENDS: tuple[Backend, ...] = ("ollama", "openai", "anthropic", "nvidia", "llama", "webgpu")


class BackendError(RuntimeError):
    """Raised when a backend cannot be constructed (missing key, missing package, etc.)."""


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    return val if val else default


def get_llm(
    backend: Optional[Backend] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Return a LangChain chat model for the requested backend.

    Args:
        backend: one of `SUPPORTED_BACKENDS`. Defaults to env `LLM_BACKEND` or "ollama".
        model:   model name override. Otherwise pulled from env per backend.
        temperature: float in [0, 1]. Defaults to env `LLM_TEMPERATURE` or 0.3.

    Raises:
        BackendError: if the backend name is unknown, a required key is missing,
        or the backend's Python package is not installed.
    """
    backend = (backend or _env("LLM_BACKEND", "ollama") or "ollama").lower()  # type: ignore[assignment]
    if backend not in SUPPORTED_BACKENDS:
        raise BackendError(f"Unknown backend: {backend}. Choose from {SUPPORTED_BACKENDS}.")

    if temperature is None:
        temperature = float(_env("LLM_TEMPERATURE", "0.3") or 0.3)

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise BackendError(f"langchain-ollama not installed: {e}") from e
        return ChatOllama(
            model=model or _env("OLLAMA_MODEL", "finnav-qwen3-vl:Q4_K_M"),
            base_url=_env("OLLAMA_HOST", "http://localhost:11434"),
            temperature=temperature,
        )

    if backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise BackendError(f"langchain-openai not installed: {e}") from e
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise BackendError("OPENAI_API_KEY not set.")
        return ChatOpenAI(
            model=model or _env("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=temperature,
        )

    if backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise BackendError(f"langchain-anthropic not installed: {e}") from e
        api_key = _env("ANTHROPIC_API_KEY")
        if not api_key:
            raise BackendError("ANTHROPIC_API_KEY not set.")
        return ChatAnthropic(
            model=model or _env("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            api_key=api_key,
            temperature=temperature,
        )

    if backend == "nvidia":
        # NIM exposes an OpenAI-compatible API.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise BackendError(f"langchain-openai not installed (needed for NIM): {e}") from e
        api_key = _env("NVIDIA_API_KEY")
        if not api_key:
            raise BackendError("NVIDIA_API_KEY not set.")
        # Default to a Qwen-family member to mirror the local fine-tune.
        # Override with NIM_MODEL env var. Verified callable on integrate.api.nvidia.com.
        return ChatOpenAI(
            model=model or _env("NIM_MODEL", "qwen/qwen3-next-80b-a3b-instruct"),
            api_key=api_key,
            base_url=_env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            temperature=temperature,
        )

    if backend == "llama":
        # Local llama.cpp via langchain-community's ChatLlamaCpp.
        # Model: fine-tuned Qwen3-VL-4B GGUF + mmproj from MOH749/finnav-qwen3-VL-4b-gguf.
        try:
            from langchain_community.chat_models import ChatLlamaCpp
        except ImportError as e:
            raise BackendError(
                "langchain-community not installed (or ChatLlamaCpp not available)."
            ) from e
        try:
            import llama_cpp  # noqa: F401  — verify the underlying runtime is present
        except ImportError as e:
            raise BackendError(
                "llama-cpp-python not installed. Run: "
                ".venv/Scripts/pip install llama-cpp-python"
            ) from e

        gguf_path = _env(
            "LLAMA_GGUF_PATH",
            os.path.join("models", "finnav-gguf", "Qwen3-VL-4B-Instruct.Q4_K_M.gguf"),
        )
        if not os.path.isfile(gguf_path):
            raise BackendError(
                f"GGUF not found at {gguf_path}. Set LLAMA_GGUF_PATH or run "
                "`python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('MOH749/finnav-qwen3-VL-4b-gguf', "
                "'Qwen3-VL-4B-Instruct.Q4_K_M.gguf', local_dir='models/finnav-gguf')\"`"
            )

        return ChatLlamaCpp(
            model_path=gguf_path,
            n_ctx=int(_env("LLAMA_N_CTX", "4096") or 4096),
            n_threads=int(_env("LLAMA_N_THREADS", "0") or 0) or None,
            n_gpu_layers=int(_env("LLAMA_N_GPU_LAYERS", "0") or 0),
            temperature=temperature,
            max_tokens=int(_env("LLAMA_MAX_TOKENS", "1024") or 1024),
            verbose=False,
            stop=["<|im_start|>", "<|im_end|>"],
        )

    if backend == "webgpu":
        # Browser-only path — Flet web embeds WebLLM via JS bridge in a later phase.
        # Keeping the enum slot reserved so the UI's backend switcher doesn't need
        # special-casing once WebGPU lands.
        raise BackendError(
            "webgpu backend runs in the browser, not the Python server. "
            "Switch the UI to a backend the server can host (ollama / nvidia / openai / anthropic / llama)."
        )

    raise BackendError(f"Unhandled backend: {backend}")  # unreachable
