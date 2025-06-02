# cognitive-swarm-agents/src/api/main.py
import logging
import uuid
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware # Pour gérer les requêtes cross-origin si besoin

# Importer nos modules projet
from config.settings import settings
from config.logging_config import setup_logging
from src.graph.main_workflow import run_cognitive_swarm_v2_1, GraphState # Importer GraphState pour le type hint
from src.api.schemas import SwarmQueryRequest, SwarmResponse, ErrorResponse, SwarmOutputMessage

# Configurer le logging pour l'API
setup_logging(level="INFO" if not settings.DEBUG else "DEBUG")
logger = logging.getLogger("api_main")

# Initialisation de l'application FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME + " API",
    description="API to interact with the Cognitive Swarm multi-agent system.",
    version="0.1.0",
    # openapi_url=f"{settings.API_V1_STR}/openapi.json" # Si API_V1_STR est utilisé pour préfixer les routes
)

# Configuration CORS (Cross-Origin Resource Sharing) - optionnel
# Permet à des frontends hébergés sur d'autres domaines d'appeler cette API.
# À configurer avec précaution pour la production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ou spécifier les domaines autorisés ex: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"], # Ou spécifier les méthodes ex: ["GET", "POST"]
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup...")
    # Vérifier les dépendances critiques au démarrage, ex: clés API
    if not settings.OPENAI_API_KEY:
        logger.error("CRITICAL: OPENAI_API_KEY not configured. Swarm functionality will be impaired.")
    if not settings.MONGODB_URI:
        logger.error("CRITICAL: MONGODB_URI not configured. Swarm checkpointing and RAG will be impaired.")
    # On pourrait ajouter un ping à MongoDB ici si MongoDBManager est utilisé par le workflow au démarrage
    # (actuellement, il est initialisé dans main_workflow au moment de la compilation du graphe).


@app.post("/invoke_swarm", response_model=SwarmResponse, responses={500: {"model": ErrorResponse}})
async def invoke_swarm_endpoint(request_data: SwarmQueryRequest = Body(...)):
    """
    Receives a user query and an optional thread_id, then invokes the Cognitive Swarm workflow.
    Returns the synthesized output and other relevant information.
    """
    thread_id = request_data.thread_id if request_data.thread_id else "api_thread_" + str(uuid.uuid4())
    query = request_data.query

    logger.info(f"Received API request to invoke swarm. Query: '{query[:50]}...', Thread ID: {thread_id}")

    try:
        # Exécuter le workflow cognitif (qui est asynchrone)
        # run_cognitive_swarm_v2_1 retourne un dict qui correspond à GraphState
        final_graph_state_dict: Dict[str, Any] = await run_cognitive_swarm_v2_1(query=query, thread_id=thread_id)
        
        if not final_graph_state_dict:
            logger.error(f"Workflow execution returned None for thread_id: {thread_id}")
            raise HTTPException(status_code=500, detail="Workflow execution failed to return a state.")

        # Convertir les messages LangChain en notre schéma SwarmOutputMessage
        formatted_messages: Optional[List[SwarmOutputMessage]] = None
        if "messages" in final_graph_state_dict and isinstance(final_graph_state_dict["messages"], list):
            formatted_messages = [SwarmOutputMessage.from_langchain_message(msg) for msg in final_graph_state_dict["messages"]]

        response_data = SwarmResponse(
            thread_id=thread_id, # Utiliser le thread_id effectif
            user_query=final_graph_state_dict.get("user_query", query),
            synthesis_output=final_graph_state_dict.get("synthesis_output"),
            research_plan=final_graph_state_dict.get("research_plan"),
            full_message_history=formatted_messages,
            error_message=final_graph_state_dict.get("error_message"),
            final_state_keys=list(final_graph_state_dict.keys()),
            # raw_final_state=final_graph_state_dict # Optionnel: pour débogage complet
        )
        logger.info(f"Successfully processed query for thread_id: {thread_id}. Synthesis started with: {str(response_data.synthesis_output)[:50]}...")
        return response_data

    except HTTPException as http_exc:
        # Re-lever les exceptions HTTP pour que FastAPI les gère
        raise http_exc
    except Exception as e:
        logger.error(f"Error invoking Cognitive Swarm for thread_id {thread_id}: {e}", exc_info=True)
        # Retourner une réponse d'erreur générique
        # Ne pas exposer les détails de l'exception interne en production
        detail_message = "An internal error occurred while processing your request."
        if settings.DEBUG: # En mode debug, on peut être plus verbeux
            detail_message = f"An internal error occurred: {str(e)}"
        raise HTTPException(status_code=500, detail=detail_message)

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "message": f"{settings.PROJECT_NAME} API is running."}

# Pour exécuter cette API localement (nécessite uvicorn: pip install uvicorn[standard]):
# uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
#
# Puis, vous pouvez envoyer une requête POST à http://localhost:8000/invoke_swarm avec un JSON body:
# {
#   "query": "What are the latest trends in reinforcement learning for robotics?",
#   "thread_id": "optional_existing_thread_id"
# }