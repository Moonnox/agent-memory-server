import os
from enum import Enum
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings


load_dotenv()


class ModelProvider(str, Enum):
    """Type of model provider"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"


class ModelConfig(BaseModel):
    """Configuration for a model"""

    provider: ModelProvider
    name: str
    max_tokens: int
    embedding_dimensions: int = 1536  # Default for OpenAI ada-002


# Model configuration mapping
MODEL_CONFIGS = {
    # OpenAI Models
    "gpt-3.5-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-3.5-turbo",
        max_tokens=4096,
        embedding_dimensions=1536,
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-3.5-turbo-16k",
        max_tokens=16384,
        embedding_dimensions=1536,
    ),
    "gpt-4": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4",
        max_tokens=8192,
        embedding_dimensions=1536,
    ),
    "gpt-4-32k": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4-32k",
        max_tokens=32768,
        embedding_dimensions=1536,
    ),
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4o",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4o-mini",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    # Newer reasoning models
    "o1": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o1",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "o1-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o1-mini",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    "o3-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o3-mini",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Embedding models
    "text-embedding-ada-002": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-ada-002",
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
    "text-embedding-3-small": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-3-small",
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
    "text-embedding-3-large": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-3-large",
        max_tokens=8191,
        embedding_dimensions=3072,
    ),
    # Anthropic Models
    "claude-3-opus-20240229": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-opus-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-sonnet-20240229": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-sonnet-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-haiku-20240307": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-haiku-20240307",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-20240620": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20240620",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Latest Anthropic Models
    "claude-3-7-sonnet-20250219": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-7-sonnet-20250219",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-20241022": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-haiku-20241022": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-haiku-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Convenience aliases
    "claude-3-7-sonnet-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-7-sonnet-20250219",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-haiku-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-haiku-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-opus-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-opus-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
}


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    long_term_memory: bool = True
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_base: str | None = None
    anthropic_api_base: str | None = None
    generation_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # PostgreSQL/Supabase configuration (for PGVector backend)
    # Connection string format: postgresql+asyncpg://user:password@host:port/database
    postgres_url: str | None = None

    # Model selection for query optimization
    slow_model: str = "gpt-4o"  # Slower, more capable model for complex tasks
    fast_model: str = (
        "gpt-4o-mini"  # Faster, smaller model for quick tasks like query optimization
    )
    port: int = 8000
    mcp_port: int = 9000

    # Vector store factory configuration
    # Python dotted path to function that returns VectorStore or VectorStoreAdapter
    # Function signature: (embeddings: Embeddings) -> Union[VectorStore, VectorStoreAdapter]
    # Examples:
    #   - "agent_memory_server.vectorstore_factory.create_redis_vectorstore"
    #   - "my_module.my_vectorstore_factory"
    #   - "my_package.adapters.create_custom_adapter"
    vectorstore_factory: str = (
        "agent_memory_server.vectorstore_factory.create_redis_vectorstore"
    )

    # RedisVL configuration (used by default Redis factory)
    redisvl_index_name: str = "memory_records"

    # The server indexes messages in long-term memory by default. If this
    # setting is enabled, we also extract discrete memories from message text
    # and save them as separate long-term memory records.
    enable_discrete_memory_extraction: bool = True

    # Minimum importance score (0.0-1.0) for extracted memories.
    # Higher values = more selective extraction. Set to 0.0 to keep all memories.
    # Recommended: 0.3 for moderate filtering, 0.5+ for very selective extraction.
    min_memory_importance_score: float = 0.0

    # Topic modeling
    topic_model_source: Literal["BERTopic", "LLM"] = "LLM"
    # If using BERTopic, use a supported model, such as
    # "MaartenGr/BERTopic_Wikipedia"
    topic_model: str = "gpt-4o-mini"
    enable_topic_extraction: bool = True
    top_k_topics: int = 3

    # Used for extracting entities from text
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    enable_ner: bool = True
    index_all_messages_in_long_term_memory: bool = False

    # RedisVL Settings
    # TODO: Adapt to vector store settings
    redisvl_distance_metric: str = "COSINE"
    redisvl_vector_dimensions: str = "1536"
    redisvl_index_prefix: str = "memory_idx"
    redisvl_indexing_algorithm: str = "HNSW"

    # Docket settings
    docket_name: str = "memory-server"
    use_docket: bool = True

    # Authentication settings
    disable_auth: bool = True
    auth_mode: Literal["disabled", "token", "oauth2"] = "disabled"

    # OAuth2/JWT Authentication settings
    oauth2_issuer_url: str | None = None
    oauth2_audience: str | None = None
    oauth2_jwks_url: str | None = None
    oauth2_algorithms: list[str] = ["RS256"]

    # Token Authentication settings
    token_auth_enabled: bool = False

    # Auth0 Client Credentials (for testing and client applications)
    auth0_client_id: str | None = None
    auth0_client_secret: str | None = None

    # Vertex AI Configuration
    vertex_project: str | None = None
    vertex_location: str | None = None
    google_application_credentials: str | None = None

    # Working memory settings
    summarization_threshold: float = (
        0.7  # Fraction of context window that triggers summarization
    )

    # Query optimization settings
    query_optimization_prompt_template: str = """Transform this natural language query into an optimized version for semantic search. The goal is to make it more effective for finding semantically similar content while preserving the original intent.

Guidelines:
- Keep the core meaning and intent
- Use more specific and descriptive terms
- Remove unnecessary words like "tell me", "I want to know", "can you"
- Focus on the key concepts and topics
- Make it concise but comprehensive

Original query: {query}

Optimized query:"""
    min_optimized_query_length: int = 2

    # Other Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    default_mcp_user_id: str | None = None
    default_mcp_namespace: str | None = None

    # Forgetting settings
    forgetting_enabled: bool = False
    forgetting_every_minutes: int = 60
    forgetting_max_age_days: float | None = None
    forgetting_max_inactive_days: float | None = None
    # Keep only top N most recent (by recency score) when budget is set
    forgetting_budget_keep_top_n: int | None = None

    # Compaction settings
    compaction_every_minutes: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables

    @property
    def generation_model_config(self) -> ModelConfig | None:
        """Get configuration for the generation model."""
        return MODEL_CONFIGS.get(self.generation_model)

    @property
    def embedding_model_config(self) -> ModelConfig | None:
        """Get configuration for the embedding model."""
        return MODEL_CONFIGS.get(self.embedding_model)

    def load_yaml_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path) as f:
            return yaml.safe_load(f) or {}


settings = Settings()


# Propagate Vertex AI settings to environment variables for LiteLLM
# This ensures that if these are set via Settings (e.g. from .env or other sources),
# they are available to LiteLLM which relies on environment variables.
if settings.vertex_project and "VERTEX_PROJECT" not in os.environ:
    os.environ["VERTEX_PROJECT"] = settings.vertex_project
if settings.vertex_location and "VERTEX_LOCATION" not in os.environ:
    os.environ["VERTEX_LOCATION"] = settings.vertex_location
if settings.google_application_credentials and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials


def get_config():
    """Get configuration from environment and settings files."""
    config_data = {}

    # If REDIS_MEMORY_CONFIG is set, load config from file
    config_file = os.getenv("REDIS_MEMORY_CONFIG")
    if config_file:
        try:
            with open(config_file) as f:
                if config_file.endswith((".yaml", ".yml")):
                    config_data = yaml.safe_load(f) or {}
                else:
                    # Assume JSON
                    import json

                    config_data = json.load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_file} not found")
        except Exception as e:
            print(f"Warning: Error loading config file {config_file}: {e}")

    # Environment variables override file config
    for key, value in os.environ.items():
        if key.startswith("REDIS_MEMORY_"):
            config_key = key[13:].lower()  # Remove REDIS_MEMORY_ prefix
            config_data[config_key] = value

    return config_data
