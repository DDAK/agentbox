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
"""

import os
import warnings
import argparse

# Filter warnings
warnings.filterwarnings('ignore')

from agent import CodingAgent, create_agent
from lib.logger import logger


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

    args = parser.parse_args()

    # Handle cleanup command
    if args.cleanup:
        logger.info("Clearing all running sandboxes...")
        CodingAgent.clear_all_sandboxes()
        logger.info("Done!")
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
                print("\nOr type any request for the AI agent.")
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
