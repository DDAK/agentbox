"""
Model configuration and provider mappings for LiteLLM.

This module provides a registry of supported models and their configurations,
enabling the coding agent to work with multiple LLM providers.
"""

from dataclasses import dataclass
from typing import Literal, Optional

Provider = Literal["openai", "anthropic", "gemini"]


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: Provider
    litellm_model: str  # LiteLLM model identifier
    supports_tool_calling: bool
    context_window: int
    system_role: str  # "system" for most, "developer" for OpenAI
    compression_model: Optional[str] = None  # Model to use for compression


# Model registry mapping user-facing names to configurations
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # OpenAI models
    "gpt-4.1-mini": ModelConfig(
        provider="openai",
        litellm_model="gpt-4.1-mini",
        supports_tool_calling=True,
        context_window=128000,
        system_role="developer",
        compression_model="gpt-4o-mini",
    ),
    "gpt-4o": ModelConfig(
        provider="openai",
        litellm_model="gpt-4o",
        supports_tool_calling=True,
        context_window=128000,
        system_role="developer",
        compression_model="gpt-4o-mini",
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        litellm_model="gpt-4o-mini",
        supports_tool_calling=True,
        context_window=128000,
        system_role="developer",
        compression_model="gpt-4o-mini",
    ),
    "gpt-5-mini": ModelConfig(
        provider="openai",
        litellm_model="gpt-5-mini",
        supports_tool_calling=True,
        context_window=128000,
        system_role="developer",
        compression_model="gpt-5-nano",
    ),
    "gpt-5-nano": ModelConfig(
        provider="openai",
        litellm_model="gpt-5-nano",
        supports_tool_calling=True,
        context_window=128000,
        system_role="developer",
        compression_model="gpt-5-nano",
    ),
    # Anthropic models - check https://docs.anthropic.com/en/docs/about-claude/models for latest IDs
    "claude-3-5-sonnet": ModelConfig(
        provider="anthropic",
        litellm_model="claude-3-5-sonnet-20241022",
        supports_tool_calling=True,
        context_window=200000,
        system_role="system",
        compression_model="claude-3-5-haiku-20241022",
    ),
    "claude-3-5-haiku": ModelConfig(
        provider="anthropic",
        litellm_model="claude-3-5-haiku-20241022",
        supports_tool_calling=True,
        context_window=200000,
        system_role="system",
        compression_model="claude-3-5-haiku-20241022",
    ),
    "claude-3-opus": ModelConfig(
        provider="anthropic",
        litellm_model="claude-3-opus-20240229",
        supports_tool_calling=True,
        context_window=200000,
        system_role="system",
        compression_model="claude-3-5-haiku-20241022",
    ),
    # Claude Sonnet 4 (newest as of 2025)
    "claude-sonnet-4": ModelConfig(
        provider="anthropic",
        litellm_model="claude-sonnet-4-20250514",
        supports_tool_calling=True,
        context_window=200000,
        system_role="system",
        compression_model="claude-3-5-haiku-20241022",
    ),
    # Google Gemini models
    "gemini-1.5-pro": ModelConfig(
        provider="gemini",
        litellm_model="gemini/gemini-1.5-pro",
        supports_tool_calling=True,
        context_window=1000000,
        system_role="system",
        compression_model="gemini/gemini-1.5-flash",
    ),
    "gemini-1.5-flash": ModelConfig(
        provider="gemini",
        litellm_model="gemini/gemini-1.5-flash",
        supports_tool_calling=True,
        context_window=1000000,
        system_role="system",
        compression_model="gemini/gemini-1.5-flash",
    ),
    "gemini-2.0-flash": ModelConfig(
        provider="gemini",
        litellm_model="gemini/gemini-2.0-flash-exp",
        supports_tool_calling=True,
        context_window=1000000,
        system_role="system",
        compression_model="gemini/gemini-2.0-flash-exp",
    ),
}


def get_model_config(model: str) -> ModelConfig:
    """
    Get configuration for a model, with fallback for unknown models.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro")

    Returns:
        ModelConfig for the specified model
    """
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]

    # Attempt to infer provider from model name
    if model.startswith("claude") or model.startswith("anthropic/"):
        # Strip anthropic/ prefix if present - LiteLLM handles provider routing
        litellm_model = model.replace("anthropic/", "") if model.startswith("anthropic/") else model
        return ModelConfig(
            provider="anthropic",
            litellm_model=litellm_model,
            supports_tool_calling=True,
            context_window=200000,
            system_role="system",
            compression_model="claude-3-5-haiku-20241022",
        )
    elif model.startswith("gemini") or model.startswith("gemini/"):
        litellm_model = model if model.startswith("gemini/") else f"gemini/{model}"
        return ModelConfig(
            provider="gemini",
            litellm_model=litellm_model,
            supports_tool_calling=True,
            context_window=1000000,
            system_role="system",
            compression_model="gemini/gemini-1.5-flash",
        )
    else:
        # Default to OpenAI
        return ModelConfig(
            provider="openai",
            litellm_model=model,
            supports_tool_calling=True,
            context_window=128000,
            system_role="developer",
            compression_model="gpt-4o-mini",
        )


def list_supported_models() -> list[str]:
    """Return a list of all supported model names."""
    return list(MODEL_REGISTRY.keys())
