"""
MAKERS API Module

This module implements the FastAPI application for the MAKERS system.
It provides endpoints for:
- /invoke_makers: Main endpoint for interacting with the MAKERS system
- /health: Health check endpoint

The API supports:
- Cross-Origin Resource Sharing (CORS)
- Request/response validation using Pydantic models
- Error handling and logging
- Configuration validation at startup
"""

import logging
import uuid
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware

from config.settings import settings
from config.logging_config import setup_logging
from src.graph.main_workflow import run_makers_v2_1, GraphState
from src.api.schemas import SwarmQueryRequest, SwarmResponse, ErrorResponse, SwarmOutputMessage

# Configure logging
setup_logging(level="INFO" if not settings.DEBUG else "DEBUG")
logger = logging.getLogger("api_main")

# Initialize FastAPI application
app = FastAPI(
    title=f"{settings.PROJECT_NAME} API",
    description="API for interacting with the MAKERS multi-agent system",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if hasattr(settings, "ALLOWED_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define HTML content for the interactive test page
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAKERS API Test Page</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input[type="text"], textarea { width: calc(100% - 22px); padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px; margin-right: 5px; }
        button:hover { background-color: #0056b3; }
        pre { background-color: #eee; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }
        .api-links a { margin-right: 15px; color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MAKERS API Test Page</h1>
        <p class="api-links">
            <a href="/docs" target="_blank">Swagger UI (/docs)</a>
            <a href="/redoc" target="_blank">ReDoc (/redoc)</a>
        </p>

        <h2>Test <code>/invoke_makers</code> (POST)</h2>
        <form id="invokeForm">
            <label for="query">Query:</label>
            <textarea id="query" name="query" rows="3" required>What are the latest advancements in using large language models for robot task planning?</textarea>
            
            <label for="thread_id">Thread ID (optional):</label>
            <input type="text" id="thread_id" name="thread_id" placeholder="e.g., api_thread_...">
            
            <button type="submit">Invoke MAKERS</button>
        </form>

        <h2>Test <code>/health</code> (GET)</h2>
        <button id="healthCheckBtn">Check Health</button>

        <h2>API Response:</h2>
        <pre id="responseArea">API responses will appear here...</pre>
    </div>

    <script>
        const responseArea = document.getElementById('responseArea');

        document.getElementById('invokeForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            const query = document.getElementById('query').value;
            const thread_id = document.getElementById('thread_id').value;
            
            responseArea.textContent = 'Loading...';
            
            try {
                const response = await fetch('/invoke_makers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, thread_id: thread_id || undefined })
                });
                const data = await response.json();
                responseArea.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                responseArea.textContent = 'Error: ' + error.message;
            }
        });

        document.getElementById('healthCheckBtn').addEventListener('click', async function() {
            responseArea.textContent = 'Loading...';
            try {
                const response = await fetch('/health');
                const data = await response.json();
                responseArea.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                responseArea.textContent = 'Error: ' + error.message;
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_interactive_page():
    return HTML_CONTENT

@app.on_event("startup")
async def startup_event():
    """
    Perform startup checks and initialization.
    Validates critical configuration and dependencies.
    """
    logger.info("FastAPI application startup...")
    
    # Check critical dependencies
    if settings.DEFAULT_LLM_MODEL_PROVIDER and settings.DEFAULT_LLM_MODEL_PROVIDER.lower() == "openai" and not settings.OPENAI_API_KEY:
        logger.error("CRITICAL: OpenAI is selected as the LLM provider, but OPENAI_API_KEY is not configured. Swarm functionality will be impaired.")
    elif not settings.OPENAI_API_KEY:
        logger.warning("WARNING: OPENAI_API_KEY is not configured. This is only an issue if you intend to use OpenAI as the LLM provider.")
    
    if not settings.MONGODB_URI:
        logger.error("CRITICAL: MONGODB_URI not configured. Swarm checkpointing and RAG will be impaired.")

@app.post(
    "/invoke_makers",
    response_model=SwarmResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        400: {"model": ErrorResponse, "description": "Bad request"}
    },
    summary="Invoke the MAKERS system",
    description="Process a user query through the MAKERS multi-agent system and return the synthesized response."
)
async def invoke_makers_endpoint(request_data: SwarmQueryRequest = Body(...)):
    """
    Process a user query through the MAKERS system.
    
    Args:
        request_data: The user query and optional thread ID
        
    Returns:
        SwarmResponse: The system's response including synthesis and metadata
        
    Raises:
        HTTPException: If the request processing fails
    """
    # Generate or use thread ID
    thread_id = request_data.thread_id if request_data.thread_id else f"api_thread_{uuid.uuid4()}"
    query = request_data.query

    logger.info(f"Received API request to invoke swarm. Query: '{query[:50]}...', Thread ID: {thread_id}")

    try:
        # Execute the cognitive workflow
        final_graph_state_dict: Dict[str, Any] = await run_makers_v2_1(
            query=query,
            thread_id=thread_id
        )
        
        if not final_graph_state_dict:
            logger.error(f"Workflow execution returned None for thread_id: {thread_id}")
            raise HTTPException(
                status_code=500,
                detail="Workflow execution failed to return a state."
            )

        # Convert LangChain messages to our schema
        formatted_messages: Optional[List[SwarmOutputMessage]] = None
        if "messages" in final_graph_state_dict and isinstance(final_graph_state_dict["messages"], list):
            formatted_messages = [
                SwarmOutputMessage.from_langchain_message(msg)
                for msg in final_graph_state_dict["messages"]
            ]

        # Construct response
        response_data = SwarmResponse(
            thread_id=thread_id,
            user_query=final_graph_state_dict.get("user_query", query),
            synthesis_output=final_graph_state_dict.get("synthesis_output"),
            research_plan=final_graph_state_dict.get("research_plan"),
            full_message_history=formatted_messages,
            error_message=final_graph_state_dict.get("error_message"),
            final_state_keys=list(final_graph_state_dict.keys())
        )
        
        logger.info(
            f"Successfully processed query for thread_id: {thread_id}. "
            f"Synthesis started with: {str(response_data.synthesis_output)[:50]}..."
        )
        return response_data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error invoking MAKERS for thread_id {thread_id}: {e}", exc_info=True)
        detail_message = (
            f"An internal error occurred: {str(e)}" if settings.DEBUG
            else "An internal error occurred while processing your request."
        )
        raise HTTPException(status_code=500, detail=detail_message)

@app.get(
    "/health",
    summary="Health check endpoint",
    description="Verify that the API is running and healthy.",
    response_model=Dict[str, str]
)
async def health_check():
    """
    Perform a health check of the API.
    
    Returns:
        Dict[str, str]: Status information
    """
    return {
        "status": "healthy",
        "message": f"{settings.PROJECT_NAME} API is running."
    }

# Development server instructions
"""
To run this API locally:
1. Install dependencies: pip install uvicorn[standard]
2. Run: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Example API request:
POST http://localhost:8000/invoke_makers
{
    "query": "What are the latest trends in reinforcement learning for robotics?",
    "thread_id": "optional_existing_thread_id"
}
"""