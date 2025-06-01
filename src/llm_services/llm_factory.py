# src/llm_services/llm_factory.py
import logging
from typing import Optional, List, Iterator, AsyncIterator, Any 

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessageChunk 
from langchain_core.outputs import ChatGenerationChunk, ChatResult 
from langchain_core.callbacks import CallbackManagerForLLMRun 
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
# MODIFICATION: Nouvel import pour ChatOllama
from langchain_ollama import ChatOllama 

from transformers import AutoTokenizer

from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_LLM_TEMPERATURE = 0.0
SYNTHESIS_LLM_TEMPERATURE = 0.5

class StreamFallbackChatHuggingFace(ChatHuggingFace):
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if isinstance(self.llm, HuggingFaceEndpoint): 
            logger.warning(
                f"HuggingFaceEndpoint (LLM: {self.llm.repo_id if hasattr(self.llm, 'repo_id') else 'N/A'}) may not fully support native streaming with all configurations. "
                f"Using non-streaming generation for {type(self).__name__}._stream and yielding as a single chunk."
            )
            chat_result: ChatResult = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for generation in chat_result.generations:
                message_chunk = AIMessageChunk(
                    content=str(generation.message.content), 
                    additional_kwargs=generation.message.additional_kwargs,
                    response_metadata=generation.message.response_metadata if hasattr(generation.message, 'response_metadata') else {}
                )
                yield ChatGenerationChunk(message=message_chunk)
            return
        else:
            logger.debug(f"Using parent _stream for LLM type: {type(self.llm)}")
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if isinstance(self.llm, HuggingFaceEndpoint):
            logger.warning(
                f"HuggingFaceEndpoint (LLM: {self.llm.repo_id if hasattr(self.llm, 'repo_id') else 'N/A'}) may not fully support native async streaming. "
                f"Using non-streaming async generation for {type(self).__name__}._astream and yielding as a single chunk."
            )
            chat_result: ChatResult = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for generation in chat_result.generations:
                message_chunk = AIMessageChunk(
                    content=str(generation.message.content), 
                    additional_kwargs=generation.message.additional_kwargs,
                    response_metadata=generation.message.response_metadata if hasattr(generation.message, 'response_metadata') else {}
                )
                yield ChatGenerationChunk(message=message_chunk)
            return
        else:
            logger.debug(f"Using parent _astream for LLM type: {type(self.llm)}")
            async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk

def get_llm(
    temperature: Optional[float] = None,
    model_provider_override: Optional[str] = None,
    model_name_override: Optional[str] = None
) -> BaseLanguageModel:
    provider = model_provider_override or settings.DEFAULT_LLM_MODEL_PROVIDER
    provider = provider.lower()
    effective_temperature = DEFAULT_LLM_TEMPERATURE if temperature is None else temperature

    logger.info(f"Initializing LLM from llm_factory for provider: '{provider}' with temperature: {effective_temperature}")

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("OpenAI API key is not configured.")
            raise ValueError("OpenAI API key is missing.")
        openai_model_name = model_name_override or settings.DEFAULT_OPENAI_GENERATIVE_MODEL
        logger.info(f"Using OpenAI model: {openai_model_name}")
        return ChatOpenAI(
            model=openai_model_name,
            temperature=effective_temperature,
            api_key=settings.OPENAI_API_KEY
        )
    elif provider == "huggingface_api":
        if not settings.HUGGINGFACE_API_KEY:
            logger.error("HuggingFace API key is not configured for Inference API.")
            raise ValueError("HuggingFace API key is missing.")
        hf_repo_id = model_name_override or settings.HUGGINGFACE_REPO_ID
        if not hf_repo_id:
            logger.error("HuggingFace Repository ID is not configured.")
            raise ValueError("HuggingFace Repository ID is missing.")
        
        logger.info(f"Creating HuggingFaceEndpoint (from langchain_huggingface) instance for repo: {hf_repo_id}")
        endpoint_llm = HuggingFaceEndpoint(
            repo_id=hf_repo_id, 
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
            temperature=effective_temperature,
            max_new_tokens=1024,
            model_kwargs={} 
        )

        try:
            logger.info(f"Loading tokenizer for {hf_repo_id} using HUGGINGFACE_API_KEY as token...")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_repo_id, 
                token=settings.HUGGINGFACE_API_KEY 
            )
            logger.info(f"Tokenizer for {hf_repo_id} loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load tokenizer for '{hf_repo_id}' using AutoTokenizer. Error: {e}", exc_info=True)
            raise ValueError(f"Failed to load tokenizer for Hugging Face model {hf_repo_id}. "
                             f"Ensure you have accepted the model's terms on Hugging Face website and your HUGGINGFACE_API_KEY is correct. "
                             f"Original error: {e}")

        logger.info(f"Wrapping HuggingFaceEndpoint LLM with StreamFallbackChatHuggingFace for provider: {provider}")
        return StreamFallbackChatHuggingFace(llm=endpoint_llm, tokenizer=tokenizer) 
        
    elif provider == "ollama":
        if not settings.OLLAMA_BASE_URL:
            logger.error("Ollama base URL is not configured.")
            raise ValueError("Ollama base URL is missing.")
        ollama_model_name = model_name_override or settings.OLLAMA_GENERATIVE_MODEL_NAME
        if not ollama_model_name:
            logger.error("Ollama generative model name is not configured.")
            raise ValueError("Ollama generative model name is missing.")
        logger.info(f"Using Ollama model (via langchain_ollama): {ollama_model_name} from {settings.OLLAMA_BASE_URL}")
        # MODIFICATION: Utilisation de ChatOllama depuis langchain_ollama
        return ChatOllama(
            model=ollama_model_name,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=effective_temperature
            # D'autres paramètres comme 'format="json"' peuvent être ajoutés ici si le modèle le supporte et que c'est nécessaire.
        )
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(level="DEBUG")
    logger.info("--- Testing LLM Factory (with updated ChatOllama import) ---")
    try:
        # settings.DEFAULT_LLM_MODEL_PROVIDER = "ollama" # Décommentez pour forcer ce test
        # settings.OLLAMA_GENERATIVE_MODEL_NAME="mistral"
        
        logger.info(f"Attempting to get LLM for provider: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
        default_llm = get_llm()
        logger.info(f"Successfully got LLM instance of type: {type(default_llm)} for provider '{settings.DEFAULT_LLM_MODEL_PROVIDER}'")
        
        if settings.DEFAULT_LLM_MODEL_PROVIDER.lower() == "ollama":
            assert isinstance(default_llm, ChatOllama), \
                f"Expected langchain_ollama.ChatOllama instance for ollama provider, but got {type(default_llm)}"
            logger.info("Assertion for langchain_ollama.ChatOllama instance passed.")
            # Test d'invocation simple si c'est Ollama
            # print(default_llm.invoke("Why is the sky blue?"))


    except ValueError as ve:
        logger.error(f"Test failed due to ValueError: {ve}. Check your .env and settings.py.")
    except ImportError as ie:
        logger.error(f"Test failed due to ImportError: {ie}. Ensure dependencies are installed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in llm_factory test: {e}", exc_info=True)
    logger.info("--- LLM Factory Test Finished ---")