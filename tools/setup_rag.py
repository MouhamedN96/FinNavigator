"""
RAG Setup Script - FinNavigator
===============================

Initializes the vector knowledge base by indexing local documents
and verifying the retrieval pipeline.

Author: Antigravity
"""

import os
import sys
import logging
from typing import List
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from tools.knowledge_tools import KnowledgeBaseIndexTool, KnowledgeBaseSearchTool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Setup")

def index_local_files():
    """Index PDF and DOCX files found in the project root"""
    root_dir = Path(__file__).parent.parent
    persist_directory = str(root_dir / "data" / "chroma_db")
    
    # Initialize Embeddings
    logger.info("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize Chroma
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="finnav_knowledge"
    )
    
    index_tool = KnowledgeBaseIndexTool(vectorstore=vectorstore)
    
    # Files to index
    files_to_index = [
        "Global Corporate Convergence_ An Analysis of Resea....docx",
        "Young Graduates' Career Moves & Strategies.pdf"
    ]
    
    for filename in files_to_index:
        file_path = root_dir / filename
        if not file_path.exists():
            logger.warning(f"File not found: {filename}")
            continue
            
        logger.info(f"Indexing {filename}...")
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(str(file_path))
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(str(file_path))
            else:
                continue
                
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            metadatas = [{"source": filename, "page": i} for i, _ in enumerate(docs)]
            
            result = index_tool._run(texts=texts, metadata=metadatas, source_type="local_doc")
            logger.info(result)
        except Exception as e:
            logger.error(f"Failed to index {filename}: {e}")

def verify_retrieval():
    """Verify that the hybrid search and reranking work"""
    root_dir = Path(__file__).parent.parent
    persist_directory = str(root_dir / "data" / "chroma_db")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="finnav_knowledge"
    )
    
    search_tool = KnowledgeBaseSearchTool(vectorstore=vectorstore, embeddings=embeddings)
    
    queries = [
        "What are the career strategies for young graduates?",
        "Global corporate convergence analysis"
    ]
    
    for query in queries:
        logger.info(f"Testing query: '{query}'")
        try:
            results = search_tool._run(query=query, top_k=3)
            logger.info(f"Results for '{query}':\n{results[:500]}...")
        except Exception as e:
            logger.error(f"Search failed: {e}")

if __name__ == "__main__":
    logger.info("Starting RAG Setup...")
    index_local_files()
    logger.info("Verifying retrieval pipeline...")
    verify_retrieval()
    logger.info("RAG Setup Complete!")
