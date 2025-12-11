# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv


def load_env():
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    load_env()
    return os.getenv("OPENAI_API_KEY")


def get_anthropic_api_key():
    load_env()
    return os.getenv("ANTHROPIC_API_KEY")


def get_google_api_key():
    load_env()
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def get_api_key_for_model(model: str) -> tuple[str, str]:
    """
    Get the appropriate API key based on the model name.

    Returns:
        Tuple of (provider, api_key)
    """
    load_env()

    # Check if model is Anthropic (Claude)
    if model.startswith("claude") or model.startswith("anthropic/"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        return "anthropic", api_key

    # Check if model is Google (Gemini)
    if model.startswith("gemini") or model.startswith("gemini/"):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set in environment")
        return "gemini", api_key

    # Default to OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return "openai", api_key


def setup_api_keys_for_litellm():
    """
    Set up all available API keys for LiteLLM.

    LiteLLM reads from environment variables directly:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY or GEMINI_API_KEY
    """
    load_env()
