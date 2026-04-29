"""
Knowledge Tools for Optimized Multimodal RAG
=============================================

Advanced tools for interacting with the vector knowledge base, 
supporting hybrid search, reranking, and visual context retrieval.

Author: Antigravity
"""

from typing import Type, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
import os
import logging

logger = logging.getLogger(__name__)

class KnowledgeSearchInput(BaseModel):
    """Input schema for knowledge base search"""
    query: str = Field(description="Search query")
    top_k: int = Field(default=5, description="Number of results to return")
    use_hybrid: bool = Field(default=True, description="Whether to use hybrid (semantic + keyword) search")
    use_rerank: bool = Field(default=True, description="Whether to use Flashrank reranking")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")

class KnowledgeIndexInput(BaseModel):
    """Input schema for knowledge base indexing"""
    texts: List[str] = Field(description="List of text documents to index")
    metadata: Optional[List[Dict[str, Any]]] = Field(default=None, description="Metadata for each document")
    source_type: str = Field(default="sec_filing", description="Type of source (sec_filing, analysis, news)")

class VisualIndexInput(BaseModel):
    """Input schema for indexing visual content"""
    image_description: str = Field(description="Detailed textual description of the image/chart")
    image_id: str = Field(description="Unique identifier or path for the image")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata (ticker, date, chart_type)")

class KnowledgeBaseSearchTool(BaseTool):
    """
    Advanced Knowledge Base Search Tool with Hybrid Search and Reranking.
    """
    name: str = "knowledge_search"
    description: str = """Search the financial knowledge base using optimized RAG.
    Combines semantic similarity with keyword matching (BM25) and Flashrank reranking.
    Best for retrieving SEC filings, previous analyses, and market research."""

    args_schema: Type[BaseModel] = KnowledgeSearchInput
    vectorstore: Any = None
    embeddings: Any = None
    reranker: Any = None
    
    def __init__(self, vectorstore=None, embeddings=None, **kwargs):
        super().__init__(vectorstore=vectorstore, embeddings=embeddings, **kwargs)

    def _get_retriever(self, top_k: int = 5, use_hybrid: bool = True, use_rerank: bool = True):
        """Construct the optimized retrieval pipeline"""
        if not self.vectorstore:
            return None

        # 1. Base Semantic Retriever
        semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k * 2})

        # 2. Hybrid Retriever (Semantic + BM25)
        if use_hybrid:
            try:
                # We need all docs in the vectorstore to initialize BM25
                # Note: In a production app with millions of docs, we'd use a persistent BM25 index
                all_docs = self.vectorstore.get()["documents"]
                if all_docs:
                    bm25_retriever = BM25Retriever.from_texts(all_docs)
                    bm25_retriever.k = top_k * 2
                    
                    retriever = EnsembleRetriever(
                        retrievers=[semantic_retriever, bm25_retriever],
                        weights=[0.7, 0.3]
                    )
                else:
                    retriever = semantic_retriever
            except Exception as e:
                logger.warning(f"Failed to initialize BM25: {e}")
                retriever = semantic_retriever
        else:
            retriever = semantic_retriever

        # 3. Reranking with Flashrank
        if use_rerank:
            try:
                compressor = FlashrankRerank(top_n=top_k)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=retriever
                )
                return compression_retriever
            except Exception as e:
                logger.warning(f"Flashrank reranking failed to initialize: {e}")
                return retriever
        
        return retriever

    def _run(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute optimized knowledge search"""
        retriever = self._get_retriever(top_k, use_hybrid, use_rerank)
        
        if not retriever:
            return "Knowledge base not initialized."

        try:
            results = retriever.invoke(query)

            if not results:
                return "No relevant documents found in knowledge base."

            output = f"### Found {len(results)} relevant insights (Optimized Retrieval):\n\n"
            for i, doc in enumerate(results, 1):
                content = doc.page_content.strip()
                meta = doc.metadata or {}
                source = meta.get('source', 'Unknown Source')
                ticker = meta.get('ticker', 'N/A')
                
                output += f"#### Result {i} [{ticker} | {source}]\n"
                output += f"{content}\n\n"
                output += "---\n"

            return output

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

class KnowledgeBaseIndexTool(BaseTool):
    """
    Knowledge Base Indexing Tool with Recursive Chunking.
    """
    name: str = "knowledge_index"
    description: str = """Index new documents with recursive character splitting.
    Ensures context preservation for SEC filings and long financial reports."""

    args_schema: Type[BaseModel] = KnowledgeIndexInput
    vectorstore: Any = None
    text_splitter: Any = None

    def __init__(self, vectorstore=None, **kwargs):
        super().__init__(vectorstore=vectorstore, **kwargs)
        # Financial/SEC aware separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\nITEM ", "\nItem ", "\nPART ", "\nPart ", 
                "\n\n", "\n", " ", ""
            ]
        )

    def _run(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        source_type: str = "sec_filing",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        if not self.vectorstore:
            return "Knowledge base not initialized."

        try:
            docs = []
            for i, text in enumerate(texts):
                meta = metadata[i] if metadata else {}
                meta["source_type"] = source_type
                
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    docs.append(Document(page_content=chunk, metadata=meta))

            self.vectorstore.add_documents(docs)
            return f"Successfully indexed {len(docs)} chunks from {len(texts)} documents."
        except Exception as e:
            return f"Error indexing: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

class VisualContextIndexTool(BaseTool):
    """
    Tool for indexing visual context (descriptions of charts/images) for Multimodal RAG.
    """
    name: str = "index_visual_context"
    description: str = """Index descriptions of charts or images.
    Enables agents to retrieve 'visual memories' based on text queries."""

    args_schema: Type[BaseModel] = VisualIndexInput
    vectorstore: Any = None

    def __init__(self, vectorstore=None, **kwargs):
        super().__init__(vectorstore=vectorstore, **kwargs)

    def _run(
        self,
        image_description: str,
        image_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        if not self.vectorstore:
            return "Knowledge base not initialized."

        try:
            meta = metadata or {}
            meta.update({
                "image_id": image_id,
                "source_type": "visual_context",
                "is_visual": True
            })
            
            doc = Document(page_content=image_description, metadata=meta)
            self.vectorstore.add_documents([doc])
            
            return f"Visual context for image '{image_id}' successfully indexed."
        except Exception as e:
            return f"Error indexing visual context: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

class ContextRetrievalTool(BaseTool):
    """
    Comprehensive context retrieval tool combining text and visual memories.
    """
    name: str = "get_context"
    description: str = "Retrieve full context for a financial topic, including SEC data and visual memories."

    args_schema: Type[BaseModel] = KnowledgeSearchInput
    kb_search: KnowledgeBaseSearchTool = None

    def _run(self, query: str, **kwargs) -> str:
        if not self.kb_search:
            return "Knowledge Search tool not configured in context retrieval."
            
        return self.kb_search._run(query, **kwargs)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
