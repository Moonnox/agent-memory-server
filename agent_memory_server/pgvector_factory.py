"""PGVector/Supabase vectorstore factory.

This module provides a factory function for creating a PGVector-backed
vectorstore using Supabase (or any PostgreSQL with pgvector extension).

Usage:
    Set VECTORSTORE_FACTORY=agent_memory_server.pgvector_factory.create_pgvector_store
    Set POSTGRES_URL to your Supabase connection string
"""

import logging
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore

from agent_memory_server.config import settings
from agent_memory_server.models import MemoryRecord
from agent_memory_server.vectorstore_adapter import LangChainVectorStoreAdapter

logger = logging.getLogger(__name__)

# Table and schema configuration
TABLE_NAME = "user_agent_memories"
VECTOR_SIZE = int(settings.redisvl_vector_dimensions)  # Reuse existing config


class PGVectorStoreAdapter(LangChainVectorStoreAdapter):
    """Custom adapter for PGVector that keeps datetime objects for proper DB storage.
    
    PostgreSQL TIMESTAMP columns require actual datetime objects, not ISO strings.
    This adapter overrides memory_to_document to preserve datetime types.
    """

    def memory_to_document(self, memory: MemoryRecord) -> Document:
        """Convert a MemoryRecord to a LangChain Document with native datetime objects.
        
        Unlike the base class which converts to ISO strings, this keeps datetime
        objects for proper PostgreSQL TIMESTAMP column handling.
        """
        # Keep datetime objects as-is for PostgreSQL TIMESTAMP columns
        created_at_val = memory.created_at if memory.created_at else datetime.now(UTC)
        last_accessed_val = memory.last_accessed if memory.last_accessed else datetime.now(UTC)
        updated_at_val = memory.updated_at if memory.updated_at else datetime.now(UTC)
        persisted_at_val = memory.persisted_at  # Can be None
        event_date_val = memory.event_date  # Can be None

        pinned_int = 1 if getattr(memory, "pinned", False) else 0
        access_count_int = int(getattr(memory, "access_count", 0) or 0)

        # Serialize list fields to comma-separated strings for TEXT columns
        topics_val = memory.topics
        if isinstance(topics_val, list):
            topics_val = ",".join(topics_val)
            
        entities_val = memory.entities
        if isinstance(entities_val, list):
            entities_val = ",".join(entities_val)
            
        extracted_from_val = memory.extracted_from
        if isinstance(extracted_from_val, list):
            extracted_from_val = ",".join(extracted_from_val)

        metadata = {
            "id": memory.id,
            "id_": memory.id,
            "session_id": memory.session_id,
            "user_id": memory.user_id,
            "agent_id": memory.agent_id,
            "namespace": memory.namespace,
            "created_at": created_at_val,
            "last_accessed": last_accessed_val,
            "updated_at": updated_at_val,
            "pinned": pinned_int,
            "access_count": access_count_int,
            "topics": topics_val,
            "entities": entities_val,
            "memory_hash": memory.memory_hash,
            "discrete_memory_extracted": memory.discrete_memory_extracted,
            "memory_type": memory.memory_type.value,
            "persisted_at": persisted_at_val,
            "extracted_from": extracted_from_val,
            "event_date": event_date_val,
        }

        # Remove None values to keep metadata clean
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Document(
            page_content=memory.text,
            metadata=metadata,
        )


async def _init_pgvector_table(pg_engine: PGEngine) -> None:
    """Initialize the PGVector table if it doesn't exist.
    
    Note: For production, you should run the migration SQL manually in Supabase.
    This is provided as a convenience for development.
    """
    from langchain_postgres import Column
    from sqlalchemy.exc import ProgrammingError

    try:
        await pg_engine.ainit_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=VECTOR_SIZE,
            metadata_columns=[
                Column("id_", "TEXT"),
                Column("session_id", "TEXT"),
                Column("user_id", "TEXT"),
                Column("agent_id", "TEXT"),
                Column("namespace", "TEXT"),
                Column("created_at", "TIMESTAMP WITH TIME ZONE"),
                Column("last_accessed", "TIMESTAMP WITH TIME ZONE"),
                Column("updated_at", "TIMESTAMP WITH TIME ZONE"),
                Column("persisted_at", "TIMESTAMP WITH TIME ZONE"),
                Column("event_date", "TIMESTAMP WITH TIME ZONE"),
                Column("pinned", "INTEGER"),
                Column("access_count", "INTEGER"),
                Column("topics", "TEXT"),  # Comma-separated or JSON
                Column("entities", "TEXT"),  # Comma-separated or JSON
                Column("memory_hash", "TEXT"),
                Column("discrete_memory_extracted", "TEXT"),
                Column("memory_type", "TEXT"),
                Column("extracted_from", "TEXT"),  # Comma-separated or JSON
            ],
        )
        logger.info(f"Initialized PGVector table: {TABLE_NAME}")
    except ProgrammingError as e:
        if "already exists" in str(e).lower():
            logger.debug(f"Table {TABLE_NAME} already exists")
        else:
            raise


def create_pgvector_store(embeddings: Embeddings) -> PGVectorStoreAdapter:
    """Create a PGVector store for Supabase/PostgreSQL.
    
    This factory function creates a PGVectorStore backed by PostgreSQL with
    the pgvector extension, suitable for use with Supabase.
    
    Args:
        embeddings: Embeddings instance to use for vector generation
        
    Returns:
        VectorStoreAdapter wrapping the PGVectorStore
        
    Raises:
        ValueError: If POSTGRES_URL is not configured
        ImportError: If langchain-postgres is not installed
    """
    if not settings.postgres_url:
        raise ValueError(
            "POSTGRES_URL must be set to use PGVector backend. "
            "Example: postgresql+asyncpg://user:password@host:port/database"
        )
    
    try:
        # Create the engine with statement_cache_size=0 to support pgbouncer
        # in transaction/statement pooling mode (which doesn't support prepared statements)
        pg_engine = PGEngine.from_connection_string(
            url=settings.postgres_url,
            connect_args={"statement_cache_size": 0},
        )
        
        # Create the vectorstore synchronously using create_sync
        # This avoids async initialization issues in the factory pattern
        vectorstore = PGVectorStore.create_sync(
            engine=pg_engine,
            table_name=TABLE_NAME,
            embedding_service=embeddings,
            id_column="id",  # Match the migration's UUID column name
            content_column="content",  # Match the migration's content column
            embedding_column="embedding",  # Match the migration's vector column
            metadata_columns=[
                "id_",
                "session_id",
                "user_id", 
                "agent_id",
                "namespace",
                "created_at",
                "last_accessed",
                "updated_at",
                "persisted_at",
                "event_date",
                "pinned",
                "access_count",
                "topics",
                "entities",
                "memory_hash",
                "discrete_memory_extracted",
                "memory_type",
                "extracted_from",
            ],
        )
        
        logger.info(f"Created PGVectorStore with table: {TABLE_NAME}")
        
        # Use custom PGVector adapter that keeps datetime objects
        return PGVectorStoreAdapter(vectorstore, embeddings)
        
    except ImportError:
        logger.error(
            "langchain-postgres not installed. Install with: pip install langchain-postgres"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating PGVector store: {e}")
        raise

