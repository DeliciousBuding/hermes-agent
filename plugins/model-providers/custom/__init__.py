"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances. Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → extra_body.think = False
  - DeepSeek V4 models → thinking enabled by default.
    Opt-out via agent.reasoning_effort: none in config.yaml.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _model_supports_deepseek_thinking(model: str | None) -> bool:
    """Check if model is a DeepSeek thinking-capable model."""
    m = (model or "").strip().lower()
    if not m:
        return False
    if m.startswith("deepseek-v") and not m.startswith("deepseek-v3"):
        return True
    if m == "deepseek-reasoner":
        return True
    return False


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false, num_ctx, DeepSeek thinking."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        model: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # DeepSeek V4 thinking — default enabled (mirrors DeepSeek API).
        # Opt-out via: agent.reasoning_effort: none
        if _model_supports_deepseek_thinking(model):
            enabled = True
            effort = None
            if isinstance(reasoning_config, dict):
                if reasoning_config.get("enabled") is False:
                    enabled = False
                elif reasoning_config.get("enabled") is True:
                    enabled = True
                raw = reasoning_config.get("effort")
                effort = str(raw).strip().lower() if raw else None
                if effort == "none":
                    enabled = False

            extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

            if enabled and effort:
                if effort in ("xhigh", "max"):
                    top_level["reasoning_effort"] = "max"
                elif effort in ("low", "medium", "high"):
                    top_level["reasoning_effort"] = effort

            return extra_body, top_level

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Disable thinking when reasoning is turned off (non-DeepSeek models)
        if reasoning_config and isinstance(reasoning_config, dict):
            raw = reasoning_config.get("effort")
            _effort = str(raw).strip().lower() if raw else ""
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                extra_body["think"] = False

        return extra_body, {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not self.base_url:
            return None
        return super().fetch_models(api_key=api_key, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
)

register_provider(custom)
