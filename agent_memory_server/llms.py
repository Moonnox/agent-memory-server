import json
import logging
import os
from typing import Any

import anthropic
import litellm
import numpy as np
from openai import AsyncOpenAI

from agent_memory_server.config import (
    MODEL_CONFIGS,
    ModelConfig,
    ModelProvider,
    settings,
)


logger = logging.getLogger(__name__)


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Use LiteLLM for unknown models
    logger.info(
        f"Model {model_name} not found in static configuration, assuming LiteLLM provider"
    )
    return ModelConfig(
        provider=ModelProvider.LITELLM,
        name=model_name,
        # Default values since we can't know them for arbitrary models
        # Users should be careful with max_tokens for unknown models
        max_tokens=128000,
        embedding_dimensions=1536,
    )


class ChatMessage:
    """Unified wrapper for chat message content"""

    def __init__(self, content: str):
        self.content = content or ""


class ChatChoice:
    """Unified wrapper for a single choice in a chat response"""

    def __init__(self, message: ChatMessage | dict[str, Any] | Any):
        # Normalize message to ChatMessage for consistent access
        if isinstance(message, ChatMessage):
            self.message = message
        elif isinstance(message, dict):
            self.message = ChatMessage(message.get("content", ""))
        elif hasattr(message, "content"):
            # Object with content attribute (e.g., from OpenAI/LiteLLM)
            self.message = ChatMessage(getattr(message, "content", ""))
        else:
            self.message = ChatMessage(str(message) if message else "")


class ChatResponse:
    """Unified wrapper for chat responses from different providers"""

    def __init__(self, choices: list[Any], usage: dict[str, int]):
        self.usage = usage or {"total_tokens": 0}
        # Normalize all choices to ChatChoice objects
        self._choices: list[ChatChoice] = []
        for choice in choices or []:
            if isinstance(choice, ChatChoice):
                self._choices.append(choice)
            elif isinstance(choice, dict):
                # Dict-style choice (e.g., from Anthropic wrapper)
                msg = choice.get("message", {})
                self._choices.append(ChatChoice(msg))
            elif hasattr(choice, "message"):
                # Object-style choice (e.g., from OpenAI/LiteLLM)
                self._choices.append(ChatChoice(choice.message))
            else:
                # Fallback
                self._choices.append(ChatChoice(choice))

    @property
    def choices(self) -> list[ChatChoice]:
        return self._choices

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class AnthropicClientWrapper:
    """Wrapper for Anthropic client"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the Anthropic client"""
        anthropic_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        anthropic_api_base = base_url or os.environ.get("ANTHROPIC_API_BASE")

        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required")

        if anthropic_api_base:
            self.client = anthropic.AsyncAnthropic(
                api_key=anthropic_api_key,
                base_url=anthropic_api_base,
            )
        else:
            self.client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

    async def create_chat_completion(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, str] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: dict[str, str] | None = None,
    ) -> ChatResponse:
        """Create a chat completion using the Anthropic API"""
        try:
            # For Anthropic, we need to handle structured output differently
            if response_format and response_format.get("type") == "json_object":
                prompt = f"{prompt}\n\nYou must respond with a valid JSON object."

            if functions and function_call:
                # Add function schema to prompt
                schema = functions[0]["parameters"]
                prompt = f"{prompt}\n\nYou must respond with a JSON object matching this schema:\n{json.dumps(schema, indent=2)}"

            response = await self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            # Convert to a unified format - safely extract content
            content = ""
            if (
                hasattr(response, "content")
                and response.content
                and len(response.content) > 0
                and hasattr(response.content[0], "text")
            ):
                content = response.content[0].text

            choices = [{"message": {"content": content}}]

            # Handle both object and dictionary usage formats from API responses
            input_tokens = output_tokens = 0
            if hasattr(response, "usage"):
                if isinstance(response.usage, dict):
                    input_tokens = response.usage.get("input_tokens", 0)
                    output_tokens = response.usage.get("output_tokens", 0)
                else:
                    input_tokens = getattr(response.usage, "input_tokens", 0)
                    output_tokens = getattr(response.usage, "output_tokens", 0)

            usage = {"total_tokens": input_tokens + output_tokens}

            return ChatResponse(choices=choices, usage=usage)
        except Exception as e:
            logger.error(f"Error creating chat completion with Anthropic: {e}")
            raise

    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """
        Create embeddings for the given texts
        Note: Anthropic doesn't offer an embedding API, so we'll use OpenAI's
        embeddings or raise an error if needed
        """
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Please use OpenAI for embeddings."
        )


class OpenAIClientWrapper:
    """Wrapper for OpenAI client"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the OpenAI client based on environment variables"""

        # Regular OpenAI setup
        openai_api_base = base_url or os.environ.get("OPENAI_API_BASE")
        openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OpenAI API key is required")

        if openai_api_base:
            self.completion_client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            self.embedding_client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            self.completion_client = AsyncOpenAI(api_key=openai_api_key)
            self.embedding_client = AsyncOpenAI(api_key=openai_api_key)

    async def create_chat_completion(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, str] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: dict[str, str] | None = None,
    ) -> ChatResponse:
        """Create a chat completion using the OpenAI API"""
        try:
            # Build the request parameters
            request_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add optional parameters if provided
            if response_format:
                request_params["response_format"] = response_format
            if functions:
                request_params["functions"] = functions
            if function_call:
                request_params["function_call"] = function_call

            response = await self.completion_client.chat.completions.create(
                **request_params
            )

            # Convert to unified format
            # Handle both object and dictionary usage formats from API responses
            total_tokens = 0
            if hasattr(response, "usage"):
                if isinstance(response.usage, dict):
                    total_tokens = response.usage.get("total_tokens", 0)
                else:
                    total_tokens = getattr(response.usage, "total_tokens", 0)

            return ChatResponse(
                choices=response.choices,
                usage={"total_tokens": total_tokens},
            )
        except Exception as e:
            logger.error(f"Error creating chat completion with OpenAI: {e}")
            raise

    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """Create embeddings for the given texts"""
        try:
            embeddings = []
            embedding_model = "text-embedding-ada-002"

            # Process in batches of 20 to avoid rate limits
            batch_size = 20
            for i in range(0, len(query_vec), batch_size):
                batch = query_vec[i : i + batch_size]
                response = await self.embedding_client.embeddings.create(
                    model=embedding_model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise


class LiteLLMClientWrapper:
    """Wrapper for LiteLLM client"""

    def __init__(self, **kwargs):
        """Initialize the LiteLLM client"""
        # LiteLLM does not need initialization
        pass

    async def create_chat_completion(
        self,
        model: str,
        prompt: str,
        response_format: dict[str, str] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: dict[str, str] | None = None,
    ) -> ChatResponse:
        """Create a chat completion using LiteLLM"""
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            if response_format:
                # LiteLLM supports response_format for some providers
                kwargs["response_format"] = response_format
            if functions:
                # Map functions/tools if necessary, but LiteLLM often handles it
                # For now, pass as is, assuming LiteLLM compatibility
                kwargs["functions"] = functions
            if function_call:
                kwargs["function_call"] = function_call

            response = await litellm.acompletion(**kwargs)

            # Unified format handling
            total_tokens = 0
            if hasattr(response, "usage"):
                if isinstance(response.usage, dict):
                    total_tokens = response.usage.get("total_tokens", 0)
                else:
                    total_tokens = getattr(response.usage, "total_tokens", 0)

            return ChatResponse(
                choices=response.choices,
                usage={"total_tokens": total_tokens},
            )
        except Exception as e:
            logger.error(f"Error creating chat completion with LiteLLM: {e}")
            raise

    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """Create embeddings using LiteLLM"""
        try:
            # We don't have the model name here in arguments, which is a flaw in the interface
            # The current interface assumes the client is bound to a provider but the method
            # takes `query_vec`.
            # However, `OpenAIClientWrapper` uses a hardcoded model "text-embedding-ada-002" inside create_embedding!
            # Anthropic wrapper raises NotImplementedError.

            # We need to know which model to use.
            # But the interface is `create_embedding(self, query_vec: list[str])`.
            # The model is not passed.
            # We can use the configured embedding model from settings if available, or pass it in __init__.

            # Let's check how get_model_client is used. It's used to get a client for a specific model_name.
            # But `create_embedding` doesn't take model_name.

            # I should modify `LiteLLMClientWrapper` to accept `model_name` in `__init__` if possible,
            # or use `settings.embedding_model` if this client is intended for embeddings.

            # Wait, `get_model_client` takes `model_name`.
            # The `OpenAIClientWrapper` has `embedding_client` initialized with API keys, but `create_embedding` uses a hardcoded model "text-embedding-ada-002".
            # This seems like a bug or limitation in `OpenAIClientWrapper`.

            # If I look at `agent_memory_server/llms.py`:
            # embedding_model = "text-embedding-ada-002"
            # It ignores the `model_name` passed to `get_model_client`!

            # For LiteLLM, I should probably try to use the model name that this client was created for, if possible.
            # But `get_model_client` creates a client for a model.
            # `get_model_client(model_name)` -> returns a client wrapper.
            # So `LiteLLMClientWrapper` should probably store the `model_name` if it's going to use it for embeddings.

            # However, `OpenAIClientWrapper` doesn't store the model name.

            # Let's assume for now that if we are using LiteLLM for embeddings, we should pass the model name to `create_embedding`.
            # But the interface is fixed.

            # I will change `__init__` to accept `model_name` optionally, or just rely on `settings.embedding_model`.
            # Actually, `get_model_client` is called with a specific `model_name`.
            # If I change `get_model_client` to pass `model_name` to the wrapper constructor, that would be better.

            # For now, I'll stick to the existing pattern but maybe default to `settings.embedding_model` for embeddings.
            
            embedding_model = settings.embedding_model
            
            response = await litellm.aembedding(
                model=embedding_model,
                input=query_vec
            )
            
            embeddings = [item["embedding"] for item in response.data]
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error creating embedding with LiteLLM: {e}")
            raise


# Global LLM client cache
_model_clients = {}


# TODO: This should take a provider as input, not model name, and cache on provider
async def get_model_client(
    model_name: str,
) -> OpenAIClientWrapper | AnthropicClientWrapper | LiteLLMClientWrapper:
    """Get the appropriate client for a model using the factory.

    This is a module-level function that caches clients for reuse.

    Args:
        model_name: Name of the model to get a client for

    Returns:
        An appropriate client wrapper for the model
    """
    global _model_clients
    model = None

    if model_name not in _model_clients:
        model_config = get_model_config(model_name)

        if model_config.provider == ModelProvider.OPENAI:
            model = OpenAIClientWrapper(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base,
            )
        elif model_config.provider == ModelProvider.ANTHROPIC:
            model = AnthropicClientWrapper(
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_api_base,
            )
        elif model_config.provider == ModelProvider.LITELLM:
            model = LiteLLMClientWrapper()

        if model:
            _model_clients[model_name] = model
            return model

        raise ValueError(f"Unsupported model provider: {model_config.provider}")

    return _model_clients[model_name]


async def optimize_query_for_vector_search(
    query: str,
    model_name: str | None = None,
) -> str:
    """
    Optimize a user query for vector search using a fast model.

    This function takes a natural language query and rewrites it to be more effective
    for semantic similarity search. It uses a fast, small model to improve search
    performance while maintaining query intent.

    Args:
        query: The original user query to optimize
        model_name: Model to use for optimization (defaults to settings.fast_model)

    Returns:
        Optimized query string better suited for vector search
    """
    if not query or not query.strip():
        return query

    # Use fast model from settings if not specified
    effective_model = model_name or settings.fast_model

    # Create optimization prompt from config template
    optimization_prompt = settings.query_optimization_prompt_template.format(
        query=query
    )

    try:
        client = await get_model_client(effective_model)

        response = await client.create_chat_completion(
            model=effective_model,
            prompt=optimization_prompt,
        )

        if (
            hasattr(response, "choices")
            and response.choices
            and len(response.choices) > 0
        ):
            optimized = ""
            if hasattr(response.choices[0], "message"):
                optimized = response.choices[0].message.content
            elif hasattr(response.choices[0], "text"):
                optimized = response.choices[0].text
            else:
                optimized = str(response.choices[0])

            # Clean up the response
            optimized = optimized.strip()

            # Fallback to original if optimization failed
            if not optimized or len(optimized) < settings.min_optimized_query_length:
                logger.warning(f"Query optimization failed for: {query}")
                return query

            logger.debug(f"Optimized query: '{query}' -> '{optimized}'")
            return optimized

    except Exception as e:
        logger.warning(f"Failed to optimize query '{query}': {e}")
        # Return original query if optimization fails
        return query

    return query
