# makers/src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage # Pour typer la sortie des messages

# Schéma pour la requête d'invocation du swarm
class SwarmQueryRequest(BaseModel):
    query: str = Field(..., description="The user query/question for the MAKERS.")
    thread_id: Optional[str] = Field(None, description="Optional existing thread ID to continue a session.")
    # On pourrait ajouter d'autres paramètres de configuration ici si on veut les surcharger au runtime
    # par exemple: config_overrides: Optional[Dict[str, Any]] = None

# Schéma pour la réponse du swarm (basé sur GraphState, mais simplifié pour l'API)
# Nous allons retourner les éléments clés de GraphState
class SwarmOutputMessage(BaseModel):
    type: str
    name: Optional[str] = None
    content: Any # Le contenu peut être une chaîne ou une structure plus complexe (ex: tool_calls)

    # Permettre la création à partir d'objets BaseMessage de LangChain
    class Config:
        from_attributes = True # anciennement orm_mode = True

    @classmethod
    def from_langchain_message(cls, msg: BaseMessage):
        return cls(
            type=msg.type.upper(), 
            name=getattr(msg, 'name', None), 
            content=msg.content
        )

class SwarmResponse(BaseModel):
    thread_id: str
    user_query: str
    synthesis_output: Optional[str] = None
    research_plan: Optional[str] = None
    # On pourrait choisir de retourner plus ou moins d'informations de l'état final
    # Par exemple, les messages complets si le client veut les afficher
    full_message_history: Optional[List[SwarmOutputMessage]] = None
    error_message: Optional[str] = None
    final_state_keys: Optional[List[str]] = None # Juste pour lister les clés du dict final
    raw_final_state: Optional[Dict[str, Any]] = Field(None, description="Raw final state for debugging, structure might vary.")


# Schéma pour une réponse d'erreur générique de l'API
class ErrorResponse(BaseModel):
    detail: str