# makers/scripts/run_makers.py
import argparse
import asyncio
import logging
import uuid
from typing import Optional

from config.settings import settings
from config.logging_config import setup_logging
from src.graph.main_workflow import run_makers_v2_1

logger = logging.getLogger(__name__)

async def async_main(query: str, thread_id: Optional[str], log_level_cli: str) -> None:
    """Execute the cognitive workflow with the given query."""
    if not query:
        logger.error("Query cannot be empty")
        print("Error: Query cannot be empty. Use --query 'Your question'")
        return

    thread_id = thread_id or f"swarm_cli_thread_{uuid.uuid4()}"
    logger.info(f"Starting MAKERS with query: '{query}' (thread: {thread_id})")
    
    print(f"\n🚀 Processing: \"{query}\"")
    print(f"🧠 Thread: {thread_id}")
    print(f"⚙️ LLM: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
    print(f"⚙️ Embeddings: {settings.DEFAULT_EMBEDDING_PROVIDER}")
    print("🔄 Processing...\n")

    try:
        if settings.DEFAULT_LLM_MODEL_PROVIDER.lower() == "openai" and not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key not configured")
            print("🚨 Error: OpenAI API key not found in .env file")
            return

        final_state = await run_makers_v2_1(query, thread_id=thread_id)
        print("\n--- ✅ MAKERS Execution Complete ---")

        if not final_state:
            print("No final state returned")
            logger.error(f"No final state for thread {thread_id}")
            return

        print("\n�� Results:")
        workflow_result = final_state.get("result", {})

        if workflow_result.get("user_query"):
            print(f"  🗣️ Query: {workflow_result['user_query']}")
        if workflow_result.get("research_plan"):
            print(f"  📝 Plan: \n{workflow_result['research_plan'][:500]}...\n")

        messages = workflow_result.get("messages", [])
        if messages:
            print(f"  💬 Recent Messages:")
            for msg in messages[-5:]:
                msg_type = getattr(msg, 'type', 'UNKNOWN').upper()
                msg_name = getattr(msg, 'name', None)
                content = str(getattr(msg, 'content', 'N/A'))
                display_name = f"{msg_type} ({msg_name})" if msg_name else msg_type
                
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"    └─ {display_name}: {content[:100]}... [Tools: {len(msg.tool_calls)}]")
                elif msg_type == "TOOL":
                    tool_id = getattr(msg, 'tool_call_id', 'N/A')
                    print(f"    └─ {display_name} (ID: {tool_id}): {content[:150]}...")
                else:
                    print(f"    └─ {display_name}: {content[:150]}...")

        synthesis = final_state.get("synthesis")
        error_message = workflow_result.get("error_message")

        if synthesis:
            print("\n💡====== FINAL SYNTHESIS ======💡")
            print(synthesis)
            print("================================")
        elif error_message:
            print("\n❌====== ERROR ======❌")
            print(error_message)
            print("================================")
        else:
            print("\n🏁====== EXECUTION COMPLETE ======")
            logger.warning(f"Thread {thread_id} completed without synthesis or error message")

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}", exc_info=True)
        print(f"\n❌ Configuration error: {ve}")
        print("   Please check your .env file settings")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="MAKERS: Knowledge Discovery Engine")
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="Query to process"
    )
    parser.add_argument(
        "-t", "--thread_id",
        type=str,
        help="Optional thread ID to continue a session"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level.upper())

    try:
        asyncio.run(async_main(
            query=args.query,
            thread_id=args.thread_id,
            log_level_cli=args.log_level
        ))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted")
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        print(f"\nCritical error: {e}")

if __name__ == "__main__":
    main()