# makers/scripts/run_makers.py
import argparse
import asyncio
import logging
import uuid
from typing import Optional

from config.settings import settings
from config.logging_config import setup_logging
from src.graph.main_workflow import run_workflow

logger = logging.getLogger(__name__)

async def async_main(query: str, thread_id: Optional[str]) -> None:
    """
    Asynchronously executes the main research workflow and prints the results.
    """
    if not query:
        logger.error("The query cannot be empty.")
        print("❌ Error: Query cannot be empty. Please use --query 'Your question'")
        return

    thread_id = thread_id or f"cli_thread_{uuid.uuid4()}"

    # --- Print Header ---
    print("\n" + "=" * 60)
    print("🚀 STARTING MAKERS - KNOWLEDGE DISCOVERY ENGINE 🚀")
    print("-" * 60)
    print(f"🔹 Query: \"{query}\"")
    print(f"🔹 Thread ID: {thread_id}")
    print(f"🔹 LLM Provider: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
    print(f"🔹 Embedding Provider: {settings.DEFAULT_EMBEDDING_PROVIDER}")
    print("." * 60)
    print("\n🔄 Processing, please wait...\n")

    try:
        # Configuration check
        if settings.DEFAULT_LLM_MODEL_PROVIDER.lower() == "openai" and not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key is not configured in the .env file.")
            print("🚨 Error: OPENAI_API_KEY is not set in your .env file.")
            return

        # Corrected function call
        final_state = await run_workflow(query, thread_id=thread_id)
        print("\n--- ✅ WORKFLOW EXECUTION COMPLETE ---")

        if not final_state:
            print("No final state was returned from the workflow.")
            logger.error(f"No final state was returned for thread {thread_id}")
            return

        # --- Print Results ---
        synthesis = final_state.get("synthesis")
        error = final_state.get("error")

        if synthesis:
            print("\n" + "💡" * 20)
            print("💡 FINAL SYNTHESIS:")
            print("💡" * 20)
            print(synthesis)
        elif error:
            print("\n" + "❌" * 20)
            print("❌ AN ERROR OCCURRED:")
            print("❌" * 20)
            print(error)
        else:
            print("\n🏁 Execution finished, but no synthesis or error was found.")
            logger.warning(
                f"Thread {thread_id} completed without synthesis or error message."
            )

        print("\n" + "=" * 60)

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}", exc_info=True)
        print(f"\n❌ Configuration Error: {ve}")
        print("   Please check your .env file settings.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the workflow: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")


def main():
    """Parses command-line arguments and runs the main async function."""
    parser = argparse.ArgumentParser(
        description="MAKERS: A multi-agent framework for knowledge discovery."
    )
    parser.add_argument(
        "-q", "--query", type=str, required=True, help="The research query to process."
    )
    parser.add_argument(
        "-t",
        "--thread_id",
        type=str,
        help="An optional thread ID to resume a previous session.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level.upper())

    try:
        asyncio.run(async_main(query=args.query, thread_id=args.thread_id))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        print("\n\nProcess interrupted by user. Exiting.")
    except Exception as e:
        logger.critical(f"A critical error occurred in the main runner: {e}", exc_info=True)
        print(f"\n🚨 A critical application error occurred: {e}")


if __name__ == "__main__":
    main()