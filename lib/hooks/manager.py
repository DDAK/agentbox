"""
Hook Manager for the lifecycle hooks system.

The HookManager is responsible for registering, storing, and executing hooks
at the appropriate points in the agent lifecycle.
"""

import subprocess
import json
from typing import Callable, Optional, List, Dict, Any, Union
from datetime import datetime
from .types import HookEvent, HookContext, HookResult
from ..logger import logger


# Type alias for hook handlers
HookHandler = Callable[[HookContext], Optional[Union[HookResult, Dict[str, Any]]]]


class HookManager:
    """
    Manages lifecycle hooks for the agent.

    The HookManager provides methods to:
    - Register hooks using decorators or direct registration
    - Execute hooks at specific lifecycle events
    - Run bash or Python scripts as hooks
    - Handle hook results and modifications

    Example:
        hooks = HookManager()

        @hooks.on(HookEvent.SessionStart)
        def on_start(ctx):
            print("Session started!")

        # Or register directly
        hooks.register(HookEvent.Stop, my_handler)

        # Execute hooks
        result = hooks.trigger(HookEvent.SessionStart, context)
    """

    def __init__(self):
        """Initialize the hook manager."""
        self._hooks: Dict[HookEvent, List[HookHandler]] = {
            event: [] for event in HookEvent
        }
        self._script_hooks: Dict[HookEvent, List[Dict[str, str]]] = {
            event: [] for event in HookEvent
        }

    def on(self, event: HookEvent) -> Callable[[HookHandler], HookHandler]:
        """
        Decorator to register a hook handler for a specific event.

        Args:
            event: The hook event to listen for

        Returns:
            Decorator function

        Example:
            @hooks.on(HookEvent.PreToolUse)
            def check_command(ctx):
                if ctx.tool_name == 'bash':
                    # Check command safety
                    pass
        """
        def decorator(handler: HookHandler) -> HookHandler:
            self.register(event, handler)
            return handler
        return decorator

    def register(self, event: HookEvent, handler: HookHandler) -> None:
        """
        Register a hook handler for a specific event.

        Args:
            event: The hook event to listen for
            handler: Callable that takes HookContext and returns Optional[HookResult]
        """
        if event not in self._hooks:
            raise ValueError(f"Invalid hook event: {event}")
        self._hooks[event].append(handler)
        logger.debug(f"[hooks] Registered handler for {event.value}: {handler.__name__}")

    def register_script(
        self,
        event: HookEvent,
        script_path: str,
        script_type: str = "bash"
    ) -> None:
        """
        Register a script to run as a hook.

        Args:
            event: The hook event to listen for
            script_path: Path to the script file
            script_type: Type of script ("bash" or "python")
        """
        if script_type not in ("bash", "python"):
            raise ValueError(f"Invalid script type: {script_type}")
        self._script_hooks[event].append({
            'path': script_path,
            'type': script_type,
        })
        logger.debug(f"[hooks] Registered {script_type} script for {event.value}: {script_path}")

    def unregister(self, event: HookEvent, handler: HookHandler) -> bool:
        """
        Unregister a hook handler.

        Args:
            event: The hook event
            handler: The handler to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        if handler in self._hooks[event]:
            self._hooks[event].remove(handler)
            return True
        return False

    def clear(self, event: Optional[HookEvent] = None) -> None:
        """
        Clear all hooks for a specific event, or all hooks if event is None.

        Args:
            event: The hook event to clear, or None to clear all
        """
        if event:
            self._hooks[event] = []
            self._script_hooks[event] = []
        else:
            for e in HookEvent:
                self._hooks[e] = []
                self._script_hooks[e] = []

    def trigger(
        self,
        event: HookEvent,
        context: Optional[HookContext] = None,
        **context_kwargs
    ) -> HookResult:
        """
        Trigger all hooks for a specific event.

        Args:
            event: The hook event to trigger
            context: Optional pre-built HookContext
            **context_kwargs: Arguments to build HookContext if not provided

        Returns:
            Combined HookResult from all handlers
        """
        # Build context if not provided
        if context is None:
            context = HookContext(event=event, **context_kwargs)
        elif context.event != event:
            context.event = event

        combined_result = HookResult()

        # Execute Python handlers
        for handler in self._hooks[event]:
            try:
                result = self._execute_handler(handler, context)
                combined_result = self._merge_results(combined_result, result)
                if result and result.skip_remaining:
                    logger.debug(f"[hooks] Skipping remaining hooks for {event.value}")
                    return combined_result
            except Exception as e:
                logger.error(f"[hooks] Error in handler {handler.__name__}: {e}")

        # Execute script hooks
        for script_config in self._script_hooks[event]:
            try:
                result = self._execute_script(script_config, context)
                combined_result = self._merge_results(combined_result, result)
                if result and result.skip_remaining:
                    return combined_result
            except Exception as e:
                logger.error(f"[hooks] Error in script {script_config['path']}: {e}")

        return combined_result

    def _execute_handler(
        self,
        handler: HookHandler,
        context: HookContext
    ) -> Optional[HookResult]:
        """Execute a single hook handler."""
        result = handler(context)

        # Handle dict return for convenience
        if isinstance(result, dict):
            return HookResult(**result)
        return result

    def _execute_script(
        self,
        script_config: Dict[str, str],
        context: HookContext
    ) -> Optional[HookResult]:
        """Execute a script hook (bash or python)."""
        script_path = script_config['path']
        script_type = script_config['type']

        # Prepare context as JSON for the script
        context_json = json.dumps(context.to_dict())

        if script_type == "bash":
            cmd = ["bash", script_path]
        else:  # python
            cmd = ["python", script_path]

        try:
            result = subprocess.run(
                cmd,
                input=context_json,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout for scripts
            )

            if result.returncode != 0:
                logger.warning(f"[hooks] Script {script_path} exited with code {result.returncode}")
                if result.stderr:
                    logger.warning(f"[hooks] Script stderr: {result.stderr}")

            # Parse JSON output if any
            if result.stdout.strip():
                try:
                    output = json.loads(result.stdout.strip())
                    return HookResult(**output)
                except json.JSONDecodeError:
                    logger.debug(f"[hooks] Script output (not JSON): {result.stdout.strip()}")

            return None

        except subprocess.TimeoutExpired:
            logger.error(f"[hooks] Script {script_path} timed out")
            return None
        except FileNotFoundError:
            logger.error(f"[hooks] Script not found: {script_path}")
            return None

    def _merge_results(
        self,
        existing: HookResult,
        new: Optional[HookResult]
    ) -> HookResult:
        """Merge two hook results, with new result taking precedence."""
        if new is None:
            return existing

        # Block takes precedence
        if new.block:
            existing.block = True
            existing.reason = new.reason or existing.reason

        # Merge modifications
        if new.modified_arguments:
            if existing.modified_arguments:
                existing.modified_arguments.update(new.modified_arguments)
            else:
                existing.modified_arguments = new.modified_arguments

        if new.modified_result is not None:
            existing.modified_result = new.modified_result

        # Merge metadata
        existing.metadata.update(new.metadata)

        return existing

    def has_hooks(self, event: HookEvent) -> bool:
        """Check if any hooks are registered for an event."""
        return bool(self._hooks[event]) or bool(self._script_hooks[event])

    def list_hooks(self, event: Optional[HookEvent] = None) -> Dict[str, List[str]]:
        """
        List all registered hooks.

        Args:
            event: Optional event to filter by

        Returns:
            Dictionary of event -> list of handler names/paths
        """
        result = {}
        events = [event] if event else list(HookEvent)

        for e in events:
            handlers = [h.__name__ for h in self._hooks[e]]
            scripts = [s['path'] for s in self._script_hooks[e]]
            if handlers or scripts:
                result[e.value] = handlers + scripts

        return result
