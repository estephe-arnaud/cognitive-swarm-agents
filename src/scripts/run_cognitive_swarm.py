# cognitive-swarm-agents/scripts/run_cognitive_swarm.py
import argparse
import asyncio
import logging
import uuid # Pour générer un thread_id si non fourni
from typing import Optional

from config.settings import settings
from config.logging_config import setup_logging
from src.graph.main_workflow import run_cognitive_swarm_v2_1 # Assurez-vous que c'est le nom correct de la fonction

# Configurer le logger pour ce script
# setup_logging() # Sera appelé dans main()
logger = logging.getLogger(__name__)

async def async_main(query: str, thread_id: Optional[str], log_level_cli: str):
    """
    Fonction asynchrone principale pour exécuter le workflow cognitif.
    """
    # Le setup_logging est déjà fait dans main() avant d'appeler async_main
    # mais on peut le réaffirmer si on veut s'assurer du niveau pour les logs de ce module.
    # setup_logging(level=log_level_cli.upper()) # Redondant si déjà fait

    if not query:
        logger.error("Query cannot be empty.")
        print("Error: Query cannot be empty. Use --query 'Your question'.")
        return

    current_thread_id = thread_id if thread_id else "swarm_cli_thread_" + str(uuid.uuid4())
    logger.info(f"Initiating Cognitive Swarm with query: '{query}' for thread ID: {current_thread_id}")
    print(f"\n🚀 Starting Cognitive Swarm for query: \"{query}\"")
    print(f"🧠 Thread ID: {current_thread_id}\n")
    print("🔄 Processing your query, please wait...\n")

    try:
        # Vérifier si les clés API nécessaires sont présentes (surtout OpenAI pour les agents LLM)
        if not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key is not configured in settings. Agents will likely fail.")
            print("🚨 ERROR: OpenAI API key is not configured. Please set it in your .env file.")
            return

        # Exécuter le workflow
        final_state = await run_cognitive_swarm_v2_1(query, thread_id=current_thread_id)

        print("\n\n--- ✅ Cognitive Swarm Execution Finished ---")
        if final_state:
            print("\n📊 Final Graph State Summary:")
            # Afficher les éléments clés de l'état final de manière lisible
            if final_state.get("user_query"):
                print(f"  🗣️ Original Query: {final_state['user_query']}")
            if final_state.get("research_plan"):
                print(f"  📝 Research Plan: \n{final_state['research_plan'][:500]}...\n") # Afficher un extrait
            
            # Afficher les derniers messages pour voir le flux de la conversation
            messages = final_state.get("messages", [])
            if messages:
                print(f"  💬 Message History (last {min(5, len(messages))} messages):")
                for msg in messages[-5:]:
                    msg_type = getattr(msg, 'type', 'UNKNOWN_MSG_TYPE').upper()
                    msg_name = getattr(msg, 'name', None)
                    msg_content_str = str(getattr(msg, 'content', 'N/A'))
                    display_name = f"{msg_type} ({msg_name})" if msg_name else msg_type
                    
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"    └─ {display_name}: {msg_content_str[:100]}... [Tool Calls: {len(msg.tool_calls)}]")
                    elif msg_type == "TOOL": # ToolMessage
                        tool_call_id = getattr(msg, 'tool_call_id', 'N/A')
                        print(f"    └─ {display_name} (ID: {tool_call_id}): {msg_content_str[:150]}...")
                    else:
                        print(f"    └─ {display_name}: {msg_content_str[:150]}...")
            
            if final_state.get("synthesis_output"):
                print("\n\n💡====== FINAL SYNTHESIS ======💡")
                print(final_state["synthesis_output"])
                print("================================")
            elif final_state.get("error_message"):
                print(f"\n\n❌====== EXECUTION ERROR ======❌")
                print(final_state["error_message"])
                print("================================")
            else:
                print("\n\n🏁====== EXECUTION COMPLETED (No explicit synthesis output or error found in final state fields) ======")
                logger.warning(f"Execution completed for thread {current_thread_id} but no 'synthesis_output' or 'error_message' was found in the final state.")
        
        else:
            print("Cognitive Swarm execution did not return a final state.")
            logger.error(f"No final state returned for thread {current_thread_id}.")

    except Exception as e:
        logger.error(f"An unexpected error occurred while running the Cognitive Swarm: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Cognitive Swarm: Knowledge Discovery Engine CLI.")
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="The user query/question for the Cognitive Swarm to process."
    )
    parser.add_argument(
        "-t", "--thread_id",
        type=str,
        default=None,
        help="Optional existing thread ID to continue a session. A new one is generated if not provided."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )

    args = parser.parse_args()

    # Configurer le logging avec le niveau potentiellement surchargé par l'argument CLI
    # Il est important de le faire avant d'appeler toute fonction qui pourrait logger, y compris async_main.
    setup_logging(level=args.log_level.upper())

    # Exécuter la fonction asynchrone principale
    try:
        asyncio.run(async_main(query=args.query, thread_id=args.thread_id, log_level_cli=args.log_level))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user (KeyboardInterrupt).")
        print("\nProcess interrupted by user.")
    except Exception as e: # Catch-all for unexpected errors during asyncio.run or setup
        logger.critical(f"A critical error occurred in the main execution block: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}")

if __name__ == "__main__":
    # Pour exécuter ce script:
    # python -m scripts.run_cognitive_swarm --query "What are the main challenges in multi-robot coordination?" --log_level DEBUG
    # (Assurez-vous d'être à la racine du projet `cognitive-swarm-agents/` et que les dépendances sont installées)
    # Et que votre .env est configuré avec les clés API.
    main()