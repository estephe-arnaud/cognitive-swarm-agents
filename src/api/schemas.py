# makers/src/api/schemas.py
"""
API Schemas Module

This module defines the Pydantic models used for request/response validation in the MAKERS API.
It includes schemas for:
- SwarmQueryRequest: Input request for invoking the MAKERS system
- SwarmOutputMessage: Message format for agent outputs
- SwarmResponse: Complete response from the MAKERS system
- ErrorResponse: Standard error response format
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage # Pour typer la sortie des messages

# Schéma pour la requête d'invocation du swarm
class SwarmQueryRequest(BaseModel):
    """
    Request schema for invoking the MAKERS system.
    
    Attributes:
        query: The user's question or request
        thread_id: Optional ID to continue an existing conversation
    """
    query: str = Field(
        ...,
        description="The user query/question for the MAKERS system",
        min_length=1,
        max_length=1000
    )
    thread_id: Optional[str] = Field(
        None,
        description="Optional existing thread ID to continue a session",
        pattern=r"^[a-zA-Z0-9_-]+$"
    )
    # On pourrait ajouter d'autres paramètres de configuration ici si on veut les surcharger au runtime
    # par exemple: config_overrides: Optional[Dict[str, Any]] = None

# Schéma pour la réponse du swarm (basé sur GraphState, mais simplifié pour l'API)
# Nous allons retourner les éléments clés de GraphState
class SwarmOutputMessage(BaseModel):
    """
    Message format for agent outputs in the MAKERS system.
    
    Attributes:
        type: The type of message (e.g., HUMAN, AI, SYSTEM)
        name: Optional name of the agent or system component
        content: The actual message content
    """
    type: str = Field(..., description="Type of message (e.g., HUMAN, AI, SYSTEM)")
    name: Optional[str] = Field(None, description="Name of the agent or system component")
    content: Any = Field(..., description="Message content (string or structured data)")

    # Permettre la création à partir d'objets BaseMessage de LangChain
    class Config:
        """Pydantic model configuration."""
        from_attributes = True # anciennement orm_mode = True

    @classmethod
    def from_langchain_message(cls, msg: BaseMessage) -> "SwarmOutputMessage":
        """
        Create a SwarmOutputMessage from a LangChain BaseMessage.
        
        Args:
            msg: The LangChain message to convert
            
        Returns:
            SwarmOutputMessage: The converted message
        """
        return cls(
            type=msg.type.upper(), 
            name=getattr(msg, 'name', None), 
            content=msg.content
        )

class SwarmResponse(BaseModel):
    """
    Complete response from the MAKERS system.
    
    Attributes:
        thread_id: Unique identifier for the conversation
        user_query: Original user query
        synthesis_output: Final synthesized response
        research_plan: Plan generated for research
        full_message_history: Complete conversation history
        error_message: Any error that occurred
        final_state_keys: Keys present in the final state
        raw_final_state: Complete raw state (for debugging)
    """
    thread_id: str = Field(..., description="Unique identifier for the conversation")
    user_query: str = Field(..., description="Original user query")
    synthesis_output: Optional[str] = Field(None, description="Final synthesized response")
    research_plan: Optional[str] = Field(None, description="Plan generated for research")
    # On pourrait choisir de retourner plus ou moins d'informations de l'état final
    # Par exemple, les messages complets si le client veut les afficher
    full_message_history: Optional[List[SwarmOutputMessage]] = Field(
        None,
        description="Complete conversation history"
    )
    error_message: Optional[str] = Field(None, description="Any error that occurred")
    final_state_keys: Optional[List[str]] = Field(
        None,
        description="Keys present in the final state"
    )
    raw_final_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Complete raw state (for debugging only)"
    )


# Schéma pour une réponse d'erreur générique de l'API
class ErrorResponse(BaseModel):
    """
    Standard error response format for the API.
    
    Attributes:
        detail: Detailed error message
    """
    detail: str = Field(..., description="Detailed error message")