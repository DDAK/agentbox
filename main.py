#!/usr/bin/env python3
"""
Main entry point for the Coding Agent.

This script launches the coding agent with a Gradio UI or CLI
that allows users to interact with the agent to create and modify code.

Usage:
    python main.py                        # Launch with Gradio UI (local sandbox)
    python main.py --sandbox=local        # Explicit local sandbox (default)
    python main.py --sandbox=docker       # Use Docker sandbox
    python main.py --cli                  # Run in CLI mode (no UI)
    python main.py --working-dir=/path    # Set working directory
    python main.py --cleanup              # Clear running sandboxes

Session Management:
    python main.py --resume=SESSION_ID    # Resume a previous session
    python main.py sessions list          # List available sessions
    python main.py sessions show ID       # Show session details
    python main.py sessions delete ID     # Delete a session
"""

import os
import warnings
import argparse

# Filter warnings
warnings.filterwarnings('ignore')

from agent import CodingAgent, create_agent
from lib.logger import logger
from lib.memory import MemoryManager


def handle_sessions_command(args):
    """Handle session management commands."""
    memory_path = getattr(args, 'memory_path', '.agent_memory')
    manager = MemoryManager(storage_path=memory_path)

    if args.action == "list":
        sessions = manager.list_sessions()
        if not sessions:
            print("No sessions found.")
            return

        print(f"\n{'ID':<40} {'Status':<10} {'Step':<8} {'Updated'}")
        print("-" * 80)
        for s in sessions:
            print(f"{s.get('id', 'unknown'):<40} {s.get('status', 'unknown'):<10} {s.get('step', 0):<8} {s.get('updated', 'N/A')[:19]}")

    elif args.action == "show":
        if not args.session_id:
            print("Error: session_id required for 'show' action")
            return

        state = manager.restore_session(args.session_id)
        if not state:
            print(f"Session '{args.session_id}' not found.")
            return

        print(f"\nSession: {args.session_id}")
        print("-" * 50)
        print(f"Status: {state.get('status', 'unknown')}")
        print(f"Started: {state.get('started', 'N/A')}")
        print(f"Updated: {state.get('updated', 'N/A')}")
        print(f"Step: {state.get('step', 0)}")
        print(f"\nTask: {state.get('task', 'N/A')}")
        print(f"\nProgress:\n{state.get('progress', 'No progress recorded')}")
        print(f"\nSummary:\n{state.get('summary', 'No summary available')}")

    elif args.action == "delete":
        if not args.session_id:
            print("Error: session_id required for 'delete' action")
            return

        if manager.session_manager.delete_session(args.session_id):
            print(f"Session '{args.session_id}' deleted.")
        else:
            print(f"Session '{args.session_id}' not found.")


def main():
    parser = argparse.ArgumentParser(
        description="Coding Agent - Create and modify code with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Launch Gradio UI with local sandbox
  python main.py --sandbox=docker         # Use Docker for isolation
  python main.py --cli --sandbox=local    # Interactive CLI mode
  python main.py --working-dir=./myproject  # Work in specific directory
        """
    )
    parser.add_argument(
        "--sandbox",
        type=str,
        choices=["local", "docker"],
        default="local",
        help="Sandbox type: 'local' (default) or 'docker'"
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help="Working directory for the sandbox (default: current directory)"
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="Docker image to use (only with --sandbox=docker)"
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Don't create a public share link for the Gradio UI"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of launching the UI"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clear all running sandboxes and exit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to use (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of agent steps (default: 100)"
    )

    # Memory/Session arguments
    parser.add_argument(
        "--memory-path",
        type=str,
        default=".agent_memory",
        help="Path for persistent memory storage (default: .agent_memory)"
    )
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Disable persistent memory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Session ID to resume"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Steps between checkpoints (default: 100)"
    )

    # Session management subcommand
    subparsers = parser.add_subparsers(dest="command", help="Session management commands")
    sessions_parser = subparsers.add_parser("sessions", help="Manage sessions")
    sessions_parser.add_argument(
        "action",
        choices=["list", "show", "delete"],
        help="Session action"
    )
    sessions_parser.add_argument(
        "session_id",
        nargs="?",
        help="Session ID (for show/delete)"
    )

    args = parser.parse_args()

    # Handle cleanup command
    if args.cleanup:
        logger.info("Clearing all running sandboxes...")
        CodingAgent.clear_all_sandboxes()
        logger.info("Done!")
        return

    # Handle sessions command
    if args.command == "sessions":
        handle_sessions_command(args)
        return

    # Resolve working directory
    working_dir = args.working_dir
    if working_dir:
        working_dir = os.path.abspath(working_dir)
        if not os.path.isdir(working_dir):
            logger.error(f"Working directory does not exist: {working_dir}")
            return

    # Create the agent
    agent = create_agent(
        sandbox_type=args.sandbox,
        working_dir=working_dir,
        model=args.model,
        max_steps=args.max_steps,
        docker_image=args.docker_image,
        memory_path=args.memory_path,
        enable_persistence=not args.no_persistence,
        resume_session=args.resume,
        checkpoint_interval=args.checkpoint_interval,
    )

    logger.info(f"Sandbox type: {args.sandbox}")
    logger.info(f"Working directory: {agent.working_dir}")

    try:
        if args.cli:
            # Run in CLI mode
            run_cli_mode(agent)
        else:
            # Launch Gradio UI
            logger.info("Starting Coding Agent...")
            agent.launch_ui(share=not args.no_share)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        # Cleanup sandbox on exit
        agent.cleanup()


def run_cli_mode(agent: CodingAgent):
    """Run the agent in interactive CLI mode."""
    logger.info("Starting CLI mode...")
    logger.info("Type 'exit' or 'quit' to stop.")
    logger.info("Type 'help' for available commands.")
    logger.info("-" * 50)

    agent.setup_sandbox()
    logger.info(f"Working directory: {agent.working_dir}")
    logger.info("-" * 50)

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue

            if query.lower() in ('exit', 'quit'):
                logger.info("Goodbye!")
                break

            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  exit, quit  - Exit the agent")
                print("  help        - Show this help message")
                print("  pwd         - Show working directory")
                print("  sessions    - List available sessions")
                print("  session     - Show current session info")
                print("\nOr type any request for the AI agent.")
                continue

            if query.lower() == 'sessions':
                sessions = agent.list_sessions()
                if not sessions:
                    print("No sessions found.")
                else:
                    print(f"\n{'ID':<40} {'Status':<10} {'Step':<8}")
                    print("-" * 60)
                    for s in sessions:
                        print(f"{s.get('id', 'unknown'):<40} {s.get('status', 'unknown'):<10} {s.get('step', 0):<8}")
                continue

            if query.lower() == 'session':
                session_id = agent.get_current_session()
                if session_id:
                    print(f"Current session: {session_id}")
                else:
                    print("No active session.")
                continue

            if query.lower() == 'pwd':
                print(f"Working directory: {agent.working_dir}")
                continue

            # Run the agent with the query
            messages, usage = agent.run_with_logging(query)

        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
