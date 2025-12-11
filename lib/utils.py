"""
Utility functions for the coding agent.
"""

from typing import Optional
from .logger import logger
from .sandbox import BaseSandbox, LocalSandbox, DockerSandbox, create_sandbox as _create_sandbox


def create_sandbox(
    sandbox_type: str = "local",
    working_dir: Optional[str] = None,
    **kwargs
) -> BaseSandbox:
    """
    Create a sandbox for code execution.

    Args:
        sandbox_type: "local" or "docker" (default: "local")
        working_dir: Working directory for the sandbox
        **kwargs: Additional arguments for specific sandbox types

    Returns:
        A sandbox instance
    """
    return _create_sandbox(sandbox_type=sandbox_type, working_dir=working_dir, **kwargs)


def clear_sandboxes():
    """
    Clear/cleanup sandboxes.

    For local sandbox, this is a no-op.
    For docker, this would stop containers.
    """
    logger.info("[sandbox] Cleanup complete (no persistent sandboxes to clear)")
