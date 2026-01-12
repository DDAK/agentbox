"""
Universal AI Agent

This module provides a universal AI agent that can handle coding, research,
file operations, and more using local or Docker sandbox environments.

Features persistent memory for long-running sessions.
"""

import os
from typing import Optional, Literal

from lib.coding_agent import coding_agent, log
from lib.tools_schemas import tools_schemas
from lib.tools import tools
from lib.prompts import SYSTEM_PROMPT_WEB_DEV, SYSTEM_PROMPT_UNIVERSAL
from lib.ui import ui
from lib.utils import create_sandbox, clear_sandboxes
from lib.sandbox import BaseSandbox
from lib.logger import logger
from lib.llm_client import create_llm_client
from lib.memory import MemoryManager, MemoryHooks
from lib.coding_agent import get_hook_manager
from helper import load_env, setup_api_keys_for_litellm


# Default system prompt - universal agent with all capabilities
SYSTEM_PROMPT_DEFAULT = SYSTEM_PROMPT_UNIVERSAL


class CodingAgent:
    """
    A universal AI agent that can handle coding, research, file operations,
    and more in a local or Docker sandbox environment.

    Capabilities:
    - Research: Web search and content fetching
    - Coding: Code execution (Python, bash)
    - File Operations: Read, write, search, and modify files
    """

    def __init__(
        self,
        sandbox_type: Literal["local", "docker"] = "local",
        working_dir: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        max_steps: int = 100,
        system_prompt: Optional[str] = None,
        docker_image: Optional[str] = None,
        memory_path: str = ".agent_memory",
        enable_persistence: bool = True,
        resume_session: Optional[str] = None,
        checkpoint_interval: int = 100,
    ):
        """
        Initialize the coding agent.

        Args:
            sandbox_type: "local" or "docker" (default: "local")
            working_dir: Working directory for the sandbox (default: current directory)
            model: The LLM model to use. Supports:
                - OpenAI: "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"
                - Anthropic: "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"
                - Google: "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"
                (default: gpt-4.1-mini)
            max_steps: Maximum number of agent steps (default: 100)
            system_prompt: Custom system prompt (default: SYSTEM_PROMPT_DEFAULT)
            docker_image: Docker image to use (only for docker sandbox)
            memory_path: Path for persistent memory storage (default: ".agent_memory")
            enable_persistence: Enable persistent memory (default: True)
            resume_session: Session ID to resume (default: None)
            checkpoint_interval: Steps between checkpoints (default: 100)
        """
        # Load environment variables and set up API keys for LiteLLM
        setup_api_keys_for_litellm()

        # Initialize LLM client (supports multiple providers via LiteLLM)
        self.client = create_llm_client(model=model)

        # Store configuration
        self.sandbox_type = sandbox_type
        self.working_dir = working_dir or os.getcwd()
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt or SYSTEM_PROMPT_DEFAULT
        self.docker_image = docker_image
        self.checkpoint_interval = checkpoint_interval

        # Initialize sandbox
        self.sbx: Optional[BaseSandbox] = None
        self.messages = []

        # Initialize memory system with hook-based management
        self.memory_manager: Optional[MemoryManager] = None
        self.memory_hooks: Optional[MemoryHooks] = None
        if enable_persistence:
            self._init_memory(memory_path)
            self._init_memory_hooks()
            if resume_session:
                self._restore_session(resume_session)

    def _init_memory(self, memory_path: str):
        """Initialize the memory manager."""
        try:
            self.memory_manager = MemoryManager(
                storage_path=memory_path,
                llm_client=self.client,
            )
            logger.info(f"Memory system initialized at {memory_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize memory system: {e}")
            self.memory_manager = None

    def _init_memory_hooks(self):
        """Initialize and register memory hooks with the global hook manager."""
        if not self.memory_manager:
            return

        try:
            self.memory_hooks = MemoryHooks(
                self.memory_manager,
                checkpoint_interval=self.checkpoint_interval,
                auto_retrieve=True,
                top_k_context=10,
            )
            self.memory_hooks.register_all(get_hook_manager())
            logger.info("Memory hooks registered with hook manager")
        except Exception as e:
            logger.warning(f"Failed to initialize memory hooks: {e}")
            self.memory_hooks = None

    def _restore_session(self, session_id: str):
        """Restore a previous session."""
        if not self.memory_manager:
            logger.warning("Cannot restore session: memory system not initialized")
            return

        state = self.memory_manager.restore_session(session_id)
        if state:
            # Restore messages if available
            if "messages" in state:
                self.messages = state["messages"]
            logger.info(f"Restored session {session_id} at step {state.get('step', 0)}")
        else:
            logger.warning(f"Session {session_id} not found")

    def setup_sandbox(self) -> BaseSandbox:
        """Create and setup the sandbox environment."""
        logger.info(f"Setting up {self.sandbox_type} sandbox environment...")

        kwargs = {}
        if self.sandbox_type == "docker" and self.docker_image:
            kwargs["image"] = self.docker_image

        self.sbx = create_sandbox(
            sandbox_type=self.sandbox_type,
            working_dir=self.working_dir,
            **kwargs
        )
        return self.sbx

    def get_host_url(self, port: int = 3000) -> str:
        """Get the URL for the sandbox's web server."""
        if self.sbx is None:
            raise RuntimeError("Sandbox not initialized. Call setup_sandbox() first.")
        return f"http://{self.sbx.get_host(port)}"

    def run(self, query: str):
        """
        Run the agent with a query and return results.

        Args:
            query: The user's request/query

        Returns:
            Generator yielding (part, messages, usage) tuples
        """
        if self.sbx is None:
            self.setup_sandbox()

        return coding_agent(
            client=self.client,
            sbx=self.sbx,
            query=query,
            tools=tools,
            tools_schemas=tools_schemas,
            max_steps=self.max_steps,
            system=self.system_prompt,
            messages=self.messages,
            model=self.model,
            memory_manager=self.memory_manager,
            checkpoint_interval=self.checkpoint_interval,
        )

    def run_with_logging(self, query: str):
        """
        Run the agent with a query and log results to console.

        Args:
            query: The user's request/query

        Returns:
            Tuple of (messages, final_usage)
        """
        if self.sbx is None:
            self.setup_sandbox()

        return log(
            coding_agent,
            client=self.client,
            sbx=self.sbx,
            query=query,
            tools=tools,
            tools_schemas=tools_schemas,
            max_steps=self.max_steps,
            system=self.system_prompt,
            messages=self.messages,
            model=self.model,
            memory_manager=self.memory_manager,
            checkpoint_interval=self.checkpoint_interval,
        )

    def list_sessions(self) -> list[dict]:
        """List available sessions."""
        if not self.memory_manager:
            return []
        return self.memory_manager.list_sessions()

    def get_current_session(self) -> Optional[str]:
        """Get the current session ID."""
        if not self.memory_manager:
            return None
        return self.memory_manager.current_session_id

    def end_session(self, summary: str = "") -> Optional[str]:
        """End the current session."""
        if not self.memory_manager:
            return None
        return self.memory_manager.end_session(summary=summary)

    def launch_ui(self, share: bool = True, height: int = 800):
        """
        Launch the Gradio UI for the agent.

        Args:
            share: Whether to create a public share link (default: True)
            height: Height of the UI in pixels (default: 800)

        Returns:
            The Gradio demo object
        """
        if self.sbx is None:
            self.setup_sandbox()

        # For local/docker, we don't have a web preview
        # Set host to None to disable the browser preview
        host_url = None

        demo = ui(
            coding_agent,
            self.messages,
            host=host_url,
            client=self.client,
            sbx=self.sbx,
            max_steps=self.max_steps,
            system=self.system_prompt,
            tools=tools,
            tools_schemas=tools_schemas,
            model=self.model,
        )

        demo.launch(share=share, height=height)
        return demo

    def cleanup(self):
        """Kill the sandbox and cleanup resources."""
        # Unregister memory hooks
        if self.memory_hooks:
            try:
                self.memory_hooks.unregister_all(get_hook_manager())
                logger.info("Memory hooks unregistered.")
            except Exception as e:
                logger.warning(f"Failed to unregister memory hooks: {e}")
            self.memory_hooks = None

        if self.sbx:
            self.sbx.kill()
            logger.info("Sandbox cleaned up.")
            self.sbx = None

    @staticmethod
    def clear_all_sandboxes():
        """Clear all running sandboxes."""
        clear_sandboxes()


def create_agent(**kwargs) -> CodingAgent:
    """
    Factory function to create a CodingAgent instance.

    Args:
        **kwargs: Arguments to pass to CodingAgent constructor

    Returns:
        A configured CodingAgent instance
    """
    return CodingAgent(**kwargs)
