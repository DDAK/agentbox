"""
Sandbox implementations for code execution.

Supports two environments:
- local: Execute code directly on the local machine
- docker: Execute code in a Docker container
"""

import os
import subprocess
import tempfile
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

from .logger import logger


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str
    stderr: str
    exit_code: int
    error: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "error": self.error,
        }


class BaseSandbox(ABC):
    """Abstract base class for sandbox implementations."""

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = working_dir or os.getcwd()

    @abstractmethod
    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in the sandbox."""
        pass

    @abstractmethod
    def get_host(self, port: int) -> str:
        """Get the host URL for a given port."""
        pass

    @abstractmethod
    def kill(self):
        """Cleanup the sandbox."""
        pass

    @property
    def sandbox_id(self) -> str:
        """Return a unique identifier for this sandbox."""
        return f"{self.__class__.__name__}-{id(self)}"


class LocalSandbox(BaseSandbox):
    """
    Execute code directly on the local machine.

    WARNING: This executes code without isolation. Use with caution.
    """

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__(working_dir)
        self._processes: List[subprocess.Popen] = []
        logger.info(f"[sandbox] Local sandbox initialized at {self.working_dir}")

    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code locally."""
        try:
            if language == "python":
                result = subprocess.run(
                    ["python", "-c", code],
                    capture_output=True,
                    text=True,
                    cwd=self.working_dir,
                    timeout=300,
                )
            elif language == "bash":
                result = subprocess.run(
                    ["bash", "-c", code],
                    capture_output=True,
                    text=True,
                    cwd=self.working_dir,
                    timeout=300,
                )
            else:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Unsupported language: {language}",
                    exit_code=1,
                    error=f"Unsupported language: {language}",
                )

            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr="Execution timed out",
                exit_code=124,
                error="Execution timed out after 300 seconds",
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
                error=str(e),
            )

    def get_host(self, port: int) -> str:
        """Return localhost URL for the given port."""
        return f"localhost:{port}"

    def kill(self):
        """Cleanup any running processes."""
        for proc in self._processes:
            try:
                proc.terminate()
            except Exception:
                pass
        self._processes.clear()
        logger.info("[sandbox] Local sandbox cleaned up")


class DockerSandbox(BaseSandbox):
    """
    Execute code in a Docker container.

    Provides isolation from the host system.
    """

    DEFAULT_IMAGE = "python:3.12-slim"

    def __init__(
        self,
        working_dir: Optional[str] = None,
        image: str = None,
        container_name: Optional[str] = None,
    ):
        super().__init__(working_dir)
        self.image = image or self.DEFAULT_IMAGE
        self.container_name = container_name or f"coding-agent-{os.getpid()}"
        self._container_id: Optional[str] = None
        self._setup_container()

    def _setup_container(self):
        """Start a Docker container for code execution."""
        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not available or not running")

            # Start container with working directory mounted
            logger.info(f"[sandbox] Starting Docker container with image {self.image}")
            result = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", self.container_name,
                    "-v", f"{self.working_dir}:/workspace",
                    "-w", "/workspace",
                    "--network", "host",
                    self.image,
                    "tail", "-f", "/dev/null",  # Keep container running
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                # Container might already exist, try to start it
                subprocess.run(
                    ["docker", "start", self.container_name],
                    capture_output=True,
                )
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.Id}}", self.container_name],
                    capture_output=True,
                    text=True,
                )

            self._container_id = result.stdout.strip()
            logger.info(f"[sandbox] Docker container started: {self._container_id[:12]}")

        except FileNotFoundError:
            raise RuntimeError("Docker is not installed. Please install Docker or use --sandbox=local")
        except Exception as e:
            raise RuntimeError(f"Failed to start Docker container: {e}")

    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in the Docker container."""
        try:
            if language == "python":
                cmd = ["docker", "exec", self.container_name, "python", "-c", code]
            elif language == "bash":
                cmd = ["docker", "exec", self.container_name, "bash", "-c", code]
            else:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Unsupported language: {language}",
                    exit_code=1,
                    error=f"Unsupported language: {language}",
                )

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr="Execution timed out",
                exit_code=124,
                error="Execution timed out after 300 seconds",
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
                error=str(e),
            )

    def get_host(self, port: int) -> str:
        """Return localhost URL (using host network mode)."""
        return f"localhost:{port}"

    def kill(self):
        """Stop and remove the Docker container."""
        try:
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["docker", "rm", self.container_name],
                capture_output=True,
                timeout=30,
            )
            logger.info(f"[sandbox] Docker container {self.container_name} removed")
        except Exception as e:
            logger.warning(f"[sandbox] Failed to cleanup Docker container: {e}")


def create_sandbox(
    sandbox_type: str = "local",
    working_dir: Optional[str] = None,
    **kwargs
) -> BaseSandbox:
    """
    Factory function to create a sandbox.

    Args:
        sandbox_type: "local" or "docker"
        working_dir: Working directory for the sandbox
        **kwargs: Additional arguments for specific sandbox types

    Returns:
        A sandbox instance
    """
    if sandbox_type == "local":
        return LocalSandbox(working_dir=working_dir)
    elif sandbox_type == "docker":
        return DockerSandbox(working_dir=working_dir, **kwargs)
    else:
        raise ValueError(f"Unknown sandbox type: {sandbox_type}. Use 'local' or 'docker'")
