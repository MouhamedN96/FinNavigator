"""
Knowledge Tools for Agent Memory and Retrieval
==============================================

Tools for interacting with the vector knowledge base and managing
agent memory and context.

Author: MiniMax Agent
"""

from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import os


class KnowledgeSearchInput(BaseModel):
    """Input schema for knowledge base search"""
    query: str = Field(description="Search query")
    top_k: int = Field(default=5, description="Number of results to return")
    filter_metadata: Optional[Dict[str, str]] = Field(default=None, description="Metadata filters")


class KnowledgeIndexInput(BaseModel):
    """Input schema for knowledge base indexing"""
    texts: List[str] = Field(description="List of text documents to index")
    metadata: Optional[List[Dict[str, Any]]] = Field(default=None, description="Metadata for each document")


class VectorQueryInput(BaseModel):
    """Input schema for vector query"""
    query: str = Field(description="Query text")
    n_results: int = Field(default=3, description="Number of results")
    collection: str = Field(default="financial_docs", description="Collection name")


class KnowledgeBaseSearchTool(BaseTool):
    """
    Knowledge Base Search Tool for semantic search.

    Searches the vector database for relevant documents based on
    semantic similarity to the query. Uses embeddings for matching.

    Best for:
    - Finding related SEC filing content
    - Retrieving previous analysis and insights
    - Context-aware responses
    """

    name: str = "knowledge_search"
    description: str = """Search the financial knowledge base using semantic search.
    Finds relevant SEC filings, previous analyses, and research notes.
    Returns top-k most similar documents."""

    args_schema: Type[BaseModel] = KnowledgeSearchInput
    
    # Define as fields so Pydantic doesn't throw errors
    vectorstore: Any = None
    embeddings: Any = None

    def __init__(self, vectorstore=None, embeddings=None, **kwargs):
        super().__init__(vectorstore=vectorstore, embeddings=embeddings, **kwargs)

    def _run(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute knowledge search"""
        if not self.vectorstore:
            return "Knowledge base not initialized. Please configure vector store."

        try:
            results = self.vectorstore.similarity_search(
                query,
                k=top_k,
                filter=filter_metadata
            )

            if not results:
                return "No relevant documents found in knowledge base."

            output = f"Found {len(results)} relevant documents:\n\n"
            for i, doc in enumerate(results, 1):
                content = doc.page_content[:500]  # Limit content
                metadata = doc.metadata or {}
                output += f"{i}. Score: {metadata.get('score', 'N/A')}\n"
                output += f"   Source: {metadata.get('source', 'N/A')}\n"
                output += f"   Content: {content}...\n\n"

            return output

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    async def _arun(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(query, top_k, filter_metadata, run_manager)


class KnowledgeBaseIndexTool(BaseTool):
    """
    Knowledge Base Indexing Tool for adding documents.

    Indexes new documents into the vector database for future retrieval.
    Documents are chunked and embedded for semantic search.

    Use for:
    - Adding new SEC filings
    - Indexing research reports
    - Storing analysis results
    """

    name: str = "knowledge_index"
    description: str = """Index new documents into the financial knowledge base.
    Chunks and embeds documents for semantic search.
    Provide texts and optional metadata."""

    args_schema: Type[BaseModel] = KnowledgeIndexInput

    def __init__(self, vectorstore=None, text_splitter=None, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
        self.text_splitter = text_splitter

    def _run(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute knowledge indexing"""
        if not self.vectorstore:
            return "Knowledge base not initialized. Please configure vector store."

        try:
            from langchain.schema import Document

            # Split texts if splitter provided
            all_chunks = []
            all_metadata = []

            for i, text in enumerate(texts):
                if self.text_splitter:
                    chunks = self.text_splitter.split_text(text)
                else:
                    chunks = [text]

                for chunk in chunks:
                    all_chunks.append(chunk)
                    doc_metadata = metadata[i] if metadata else {}
                    all_metadata.append(doc_metadata)

            # Create documents
            docs = [
                Document(page_content=chunk, metadata=meta)
                for chunk, meta in zip(all_chunks, all_metadata)
            ]

            # Add to vectorstore
            self.vectorstore.add_documents(docs)

            return f"Successfully indexed {len(docs)} document chunks from {len(texts)} sources."

        except ImportError:
            return "Error: langchain required"
        except Exception as e:
            return f"Error indexing documents: {str(e)}"

    async def _arun(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(texts, metadata, run_manager)


class VectorQueryTool(BaseTool):
    """
    Direct Vector Query Tool for low-level vector operations.

    Provides direct access to the vector database for specialized queries.
    Useful for finding similar documents, concept exploration, etc.
    """

    name: str = "vector_query"
    description: str = """Direct vector database query for semantic similarity.
    Finds documents by embedding similarity.
    Returns document IDs, content, and similarity scores."""

    args_schema: Type[BaseModel] = VectorQueryInput

    def __init__(self, chroma_client=None, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.chroma_client = chroma_client
        self.embedding_model = embedding_model

    def _run(
        self,
        query: str,
        n_results: int = 3,
        collection: str = "financial_docs",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute vector query"""
        if not self.chroma_client or not self.embedding_model:
            return "Vector client not initialized. Please configure ChromaDB client."

        try:
            # Get embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Query collection
            collection_obj = self.chroma_client.get_or_create_collection(name=collection)
            results = collection_obj.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            if not results or not results.get("documents"):
                return "No matching documents found."

            output = f"Found {len(results['documents'][0])} similar documents:\n\n"
            for i, doc in enumerate(results["documents"][0]):
                score = results["distances"][0][i] if "distances" in results else "N/A"
                metadata = results["metadatas"][0][i] if "metadatas" in results else {}
                doc_id = results["ids"][0][i] if "ids" in results else f"doc_{i}"

                output += f"{i+1}. [ID: {doc_id}] Score: {score:.4f}\n"
                output += f"   {doc[:300]}...\n"
                if metadata:
                    output += f"   Metadata: {metadata}\n"
                output += "\n"

            return output

        except ImportError:
            return "Error: sentence-transformers required"
        except Exception as e:
            return f"Error querying vector store: {str(e)}"

    async def _arun(
        self,
        query: str,
        n_results: int = 3,
        collection: str = "financial_docs",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(query, n_results, collection, run_manager)


class MemorySearchInput(BaseModel):
    """Input schema for memory search"""
    query: str = Field(description="Query for memory search")
    agent_name: Optional[str] = Field(default=None, description="Filter by agent")
    limit: int = Field(default=5, description="Number of results")


class MemorySearchTool(BaseTool):
    """
    Agent Memory Search Tool.

    Searches conversation history and agent memory for previous
    interactions and accumulated knowledge.
    """

    name: str = "memory_search"
    description: str = """Search agent memory and conversation history.
    Find previous analyses, decisions, and interactions.
    Useful for context continuity across sessions."""

    args_schema: Type[BaseModel] = MemorySearchInput

    def __init__(self, memory_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.memory_manager = memory_manager

    def _run(
        self,
        query: str,
        agent_name: Optional[str] = None,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute memory search"""
        if not self.memory_manager:
            return "Memory manager not initialized."

        try:
            results = self.memory_manager.search(
                query=query,
                agent_filter=agent_name,
                limit=limit
            )

            if not results:
                return "No matching memories found."

            output = f"Found {len(results)} relevant memories:\n\n"
            for i, memory in enumerate(results, 1):
                output += f"{i}. {memory['content'][:200]}...\n"
                output += f"   Agent: {memory.get('agent', 'N/A')}\n"
                output += f"   Time: {memory.get('timestamp', 'N/A')}\n\n"

            return output

        except Exception as e:
            return f"Error searching memory: {str(e)}"

    async def _arun(
        self,
        query: str,
        agent_name: Optional[str] = None,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(query, agent_name, limit, run_manager)


class ContextRetrievalInput(BaseModel):
    """Input schema for context retrieval"""
    topic: str = Field(description="Topic to get context for")
    include_filings: bool = Field(default=True, description="Include SEC filings")
    include_analyses: bool = Field(default=True, description="Include previous analyses")


class ContextRetrievalTool(BaseTool):
    """
    Comprehensive Context Retrieval Tool.

    Gathers all relevant context for a topic from multiple sources:
    - SEC filings
    - Knowledge base
    - Memory
    - Previous analyses
    """

    name: str = "get_context"
    description: str = """Retrieve comprehensive context for a topic from multiple sources.
    Combines SEC filings, knowledge base, and memory for complete context.
    Use for research and analysis tasks."""

    args_schema: Type[BaseModel] = ContextRetrievalInput

    def __init__(self, sec_tool=None, kb_tool=None, memory_tool=None, **kwargs):
        super().__init__(**kwargs)
        self.sec_tool = sec_tool
        self.kb_tool = kb_tool
        self.memory_tool = memory_tool

    def _run(
        self,
        topic: str,
        include_filings: bool = True,
        include_analyses: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute context retrieval"""
        results = []

        # Search knowledge base
        if self.kb_tool:
            kb_result = self.kb_tool.invoke({
                "query": topic,
                "top_k": 3
            })
            results.append(("Knowledge Base", kb_result))

        # Search memory
        if self.memory_tool:
            mem_result = self.memory_tool.invoke({
                "query": topic,
                "limit": 3
            })
            results.append(("Memory", mem_result))

        # Format output
        output = f"Context for '{topic}':\n\n"
        for source, content in results:
            output += f"=== {source} ===\n{content}\n\n"

        return output if output.strip() != f"Context for '{topic}':" else "No context found."

    async def _arun(
        self,
        topic: str,
        include_filings: bool = True,
        include_analyses: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(topic, include_filings, include_analyses, run_manager)
