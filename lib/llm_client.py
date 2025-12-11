"""
LiteLLM-based LLM client abstraction that normalizes responses
to match the existing code's expectations.

This module provides a unified interface for multiple LLM providers
(OpenAI, Anthropic, Google Gemini) through LiteLLM, while maintaining
backward compatibility with the existing codebase.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import litellm

from .model_config import get_model_config, ModelConfig

# Configure LiteLLM
litellm.set_verbose = False

# litellm._turn_on_debug()

@dataclass
class ToolCall:
    """
    Normalized tool call representation.

    Matches the interface expected by existing code:
    - part.type == "function_call"
    - part.name, part.arguments, part.call_id
    - part.to_dict()
    """

    type: str = "function_call"
    name: str = ""
    arguments: str = ""
    call_id: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }


@dataclass
class TextContent:
    """
    Normalized text content representation.

    Matches the interface expected by existing code:
    - part.type == "message"
    - part.content (list of content blocks)
    - part.to_dict()
    """

    type: str = "message"
    content: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "content": self.content,
        }


@dataclass
class Usage:
    """Token usage information."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class NormalizedResponse:
    """
    Normalized response that mimics the structure expected by existing code.

    The existing code expects:
    - response.output (iterable of parts with .type, .name, .arguments, .call_id, .to_dict())
    - response.output_text (for text extraction)
    - response.usage.total_tokens
    """

    output: list
    usage: Usage

    @property
    def output_text(self) -> str:
        """Extract text content from the response."""
        for part in self.output:
            if hasattr(part, "type") and part.type == "message":
                if part.content and len(part.content) > 0:
                    return part.content[0].get("text", "")
        return ""


class LLMClient:
    """
    LiteLLM-based client that provides a unified interface for multiple providers.

    This client normalizes the response format to match what the existing
    coding_agent.py expects from the OpenAI responses API.
    """

    def __init__(self, model: str):
        """
        Initialize the LLM client.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro")
        """
        self.model = model
        self.config = get_model_config(model)

        if not self.config.supports_tool_calling:
            raise ValueError(f"Model {model} does not support native tool calling")

    def _convert_messages(self, input_messages: list[dict]) -> list[dict]:
        """
        Convert messages from the existing format to LiteLLM format.

        Handles:
        - "developer" role -> "system" role (for non-OpenAI providers)
        - function_call/function_call_output format conversion
        """
        converted = []
        pending_tool_calls = []

        for msg in input_messages:
            if "role" in msg:
                role = msg["role"]
                # Convert "developer" to appropriate system role
                if role == "developer":
                    role = self.config.system_role

                converted.append(
                    {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                )
            elif "type" in msg:
                msg_type = msg["type"]

                if msg_type == "function_call":
                    # Collect tool calls - they need to be grouped into a single assistant message
                    pending_tool_calls.append(
                        {
                            "id": msg.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": msg.get("name", ""),
                                "arguments": msg.get("arguments", "{}"),
                            },
                        }
                    )
                elif msg_type == "function_call_output":
                    # If we have pending tool calls, flush them first
                    if pending_tool_calls:
                        converted.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": pending_tool_calls,
                            }
                        )
                        pending_tool_calls = []

                    # Convert to tool response message
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.get("call_id", ""),
                            "content": msg.get("output", ""),
                        }
                    )
                elif msg_type == "message":
                    # Handle message type (text content from assistant)
                    content = msg.get("content", [])
                    if content and len(content) > 0:
                        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                        converted.append(
                            {
                                "role": "assistant",
                                "content": text,
                            }
                        )

        # Flush any remaining pending tool calls
        if pending_tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": pending_tool_calls,
                }
            )

        return converted

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """
        Convert tools from responses API format to chat.completions format.

        Input format (responses API - what we use):
            {"type": "function", "name": "...", "description": "...", "parameters": {...}}

        Output format (chat.completions - what LiteLLM expects):
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" not in tool:
                # Convert from responses format to chat.completions format
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                })
            else:
                # Already in correct format or unknown format, pass through
                converted.append(tool)
        return converted

    def _normalize_response(self, response) -> NormalizedResponse:
        """
        Normalize LiteLLM response to match existing code expectations.

        LiteLLM returns: response.choices[0].message.tool_calls / .content
        We need: response.output (list of parts with .type, .name, etc.)
        """
        output_parts = []

        message = response.choices[0].message

        # Handle text content
        if message.content:
            text_part = TextContent(
                type="message", content=[{"type": "text", "text": message.content}]
            )
            output_parts.append(text_part)

        # Handle tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_part = ToolCall(
                    type="function_call",
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    call_id=tool_call.id,
                )
                output_parts.append(tool_part)

        # Extract usage
        usage = Usage(
            total_tokens=getattr(response.usage, "total_tokens", 0),
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
            completion_tokens=getattr(response.usage, "completion_tokens", 0),
        )

        return NormalizedResponse(output=output_parts, usage=usage)

    def create(
        self,
        input: list[dict],
        tools: Optional[list[dict]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> NormalizedResponse:
        """
        Create a completion using LiteLLM.

        This method signature mimics the existing client.responses.create() calls.

        Args:
            input: List of messages in the existing format
            tools: Tool schemas in OpenAI function format
            model: Optional model override
            **kwargs: Additional parameters passed to LiteLLM

        Returns:
            NormalizedResponse matching existing code expectations
        """
        # Use provided model or default
        target_model = model or self.model
        config = get_model_config(target_model)

        # Convert messages to LiteLLM format
        messages = self._convert_messages(input)

        # Prepare LiteLLM call parameters
        call_params = {
            "model": config.litellm_model,
            "messages": messages,
        }

        # Add tools if provided (convert to LiteLLM format)
        if tools:
            call_params["tools"] = self._convert_tools(tools)

        # Merge additional kwargs (excluding 'model' and 'input' which we handle)
        for key, value in kwargs.items():
            if key not in ("model", "input"):
                call_params[key] = value

        # Make the LiteLLM call
        response = litellm.completion(**call_params)

        # Normalize and return
        return self._normalize_response(response)


class ResponsesAPI:
    """
    Wrapper that provides `client.responses.create()` interface.

    This maintains backward compatibility with existing code that calls:
        client.responses.create(model=..., input=..., tools=...)
    """

    def __init__(self, default_model: str):
        self.default_model = default_model
        self._client = LLMClient(default_model)

    def create(
        self,
        model: Optional[str] = None,
        input: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> NormalizedResponse:
        """
        Create a completion, maintaining the existing API signature.

        Args:
            model: Model to use (falls back to default)
            input: Messages in existing format
            tools: Tool schemas
            **kwargs: Additional parameters
        """
        return self._client.create(
            input=input or [], tools=tools, model=model, **kwargs
        )


def create_llm_client(model: str = "gpt-4o"):
    """
    Factory function to create an LLM client with responses API.

    Returns an object with a `.responses` attribute that has a `.create()` method,
    maintaining backward compatibility with existing code.

    Args:
        model: Default model to use

    Returns:
        Client object with .responses.create() interface
    """

    class ClientWrapper:
        def __init__(self, model: str):
            self.responses = ResponsesAPI(default_model=model)

    return ClientWrapper(model)
