"""
Language Model Factory Module

This module provides a factory for creating and configuring language model instances
from various providers (OpenAI, HuggingFace, Ollama). It includes a custom wrapper
for HuggingFace models with streaming fallback support.

The module handles:
- Provider-specific model initialization
- API key validation
- Model configuration
- Streaming support
- Error handling and logging
"""

import logging
from typing import Optional, List, Iterator, AsyncIterator, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer

from config.settings import settings

logger = logging.getLogger(__name__)

# Temperature settings for different use cases
DEFAULT_LLM_TEMPERATURE = 0.0  # For precise, deterministic responses
SYNTHESIS_LLM_TEMPERATURE = 0.5  # For more creative synthesis tasks

class StreamFallbackChatHuggingFace(ChatHuggingFace):
    """
    Custom wrapper for HuggingFace chat models that provides fallback streaming support.
    
    This class extends ChatHuggingFace to handle cases where the underlying model
    doesn't fully support native streaming. It implements fallback mechanisms for
    both synchronous and asynchronous streaming.
    """
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Handle streaming with fallback for HuggingFaceEndpoint.
        
        Args:
            messages: List of messages to process
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments for the model
            
        Returns:
            Iterator yielding chat generation chunks
        """
        if isinstance(self.llm, HuggingFaceEndpoint):
            repo_id = getattr(self.llm, 'repo_id', 'N/A')
            logger.warning(
                f"HuggingFaceEndpoint (LLM: {repo_id}) "
                "may not fully support native streaming. Using non-streaming generation."
            )
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for generation in result.generations:
                chunk = AIMessageChunk(
                    content=str(generation.message.content),
                    additional_kwargs=generation.message.additional_kwargs,
                    response_metadata=getattr(generation.message, 'response_metadata', {})
                )
                yield ChatGenerationChunk(message=chunk)
            return

        logger.debug(f"Using parent _stream for LLM type: {type(self.llm)}")
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Handle async streaming with fallback for HuggingFaceEndpoint.
        
        Args:
            messages: List of messages to process
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments for the model
            
        Returns:
            AsyncIterator yielding chat generation chunks
        """
        if isinstance(self.llm, HuggingFaceEndpoint):
            repo_id = getattr(self.llm, 'repo_id', 'N/A')
            logger.warning(
                f"HuggingFaceEndpoint (LLM: {repo_id}) "
                "may not fully support native async streaming. Using non-streaming async generation."
            )
            result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for generation in result.generations:
                chunk = AIMessageChunk(
                    content=str(generation.message.content),
                    additional_kwargs=generation.message.additional_kwargs,
                    response_metadata=getattr(generation.message, 'response_metadata', {})
                )
                yield ChatGenerationChunk(message=chunk)
            return

        logger.debug(f"Using parent _astream for LLM type: {type(self.llm)}")
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk

def get_llm(
    temperature: Optional[float] = None,
    model_provider_override: Optional[str] = None,
    model_name_override: Optional[str] = None
) -> BaseLanguageModel:
    """
    Create and configure a language model instance based on settings and overrides.
    
    Args:
        temperature: Optional temperature setting (defaults to DEFAULT_LLM_TEMPERATURE)
        model_provider_override: Optional override for the model provider
        model_name_override: Optional override for the model name
        
    Returns:
        Configured language model instance
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    provider = (model_provider_override or settings.DEFAULT_LLM_MODEL_PROVIDER).lower()
    temperature = DEFAULT_LLM_TEMPERATURE if temperature is None else temperature

    logger.info(f"Initializing LLM for provider: '{provider}' (temperature: {temperature})")

    if provider == "openai":
        return _create_openai_llm(temperature, model_name_override)
    elif provider == "huggingface_api":
        return _create_huggingface_llm(temperature, model_name_override)
    elif provider == "ollama":
        return _create_ollama_llm(temperature, model_name_override)
    
    raise ValueError(f"Unsupported LLM provider: {provider}")

def _create_openai_llm(temperature: float, model_name_override: Optional[str]) -> ChatOpenAI:
    """Create and configure an OpenAI language model."""
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is missing")
    
    model = model_name_override or settings.DEFAULT_OPENAI_GENERATIVE_MODEL
    logger.info(f"Using OpenAI model: {model}")
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY
    )

def _create_huggingface_llm(temperature: float, model_name_override: Optional[str]) -> StreamFallbackChatHuggingFace:
    """Create and configure a HuggingFace language model with streaming fallback."""
    if not settings.HUGGINGFACE_API_KEY:
        raise ValueError("HuggingFace API key is missing")
    
    repo_id = model_name_override or settings.HUGGINGFACE_REPO_ID
    if not repo_id:
        raise ValueError("HuggingFace Repository ID is missing")
    
    logger.info(f"Creating HuggingFaceEndpoint for repo: {repo_id}")
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        temperature=temperature,
        max_new_tokens=1024,
        model_kwargs={}
    )

    try:
        logger.info(f"Loading tokenizer for {repo_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            token=settings.HUGGINGFACE_API_KEY
        )
    except Exception as e:
        raise ValueError(
            f"Failed to load tokenizer for {repo_id}. "
            "Ensure you have accepted the model's terms on Hugging Face "
            f"and your API key is correct. Error: {e}"
        )

    return StreamFallbackChatHuggingFace(llm=endpoint, tokenizer=tokenizer)

def _create_ollama_llm(temperature: float, model_name_override: Optional[str]) -> ChatOllama:
    """Create and configure an Ollama language model."""
    if not settings.OLLAMA_BASE_URL:
        raise ValueError("Ollama base URL is missing")
    
    model = model_name_override or settings.OLLAMA_GENERATIVE_MODEL_NAME
    if not model:
        raise ValueError("Ollama model name is missing")
    
    logger.info(f"Using Ollama model: {model} from {settings.OLLAMA_BASE_URL}")
    return ChatOllama(
        model=model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature
    )

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="DEBUG")
    
    logger.info("Testing LLM Factory")
    try:
        logger.info(f"Getting LLM for provider: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
        llm = get_llm()
        logger.info(f"Successfully initialized {type(llm)} for {settings.DEFAULT_LLM_MODEL_PROVIDER}")
        
        if settings.DEFAULT_LLM_MODEL_PROVIDER.lower() == "ollama":
            assert isinstance(llm, ChatOllama), \
                f"Expected ChatOllama instance, got {type(llm)}"
            logger.info("ChatOllama instance verified")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    logger.info("LLM Factory test complete")