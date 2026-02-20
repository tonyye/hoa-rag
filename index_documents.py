#!/usr/bin/env python3
"""
Property Graph Index Query Script

Usage:
    python index_documents.py <document_path> [query]

If no query is provided, enters interactive chat mode.
"""

import os
import sys

import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PDFReader

from woodcrest_hills_document_parser import WoodcrestHillsDocumentParser

# Load environment variables from .env file
load_dotenv()

nest_asyncio.apply()

def create_graph_store(config):
    """Create a Neo4j property graph store."""
    return Neo4jPropertyGraphStore(
        username=config["NEO4J_USERNAME"],
        password=config["NEO4J_PASSWORD"],
        url=config["NEO4J_URL"],
    )


def create_embed_model(config):
    """Create an Ollama embedding model instance."""
    return OllamaEmbedding(
        model_name=config["EMBED_MODEL"],
        request_timeout=240.0,
        base_url=config["EMBED_BASE_URL"],
    )


def create_llm(config):
    """Create an Ollama LLM instance."""
    return Ollama(
        model=config["LLM_MODEL"],
        request_timeout=240.0,
        context_window=9216,
        base_url=config["LLM_BASE_URL"],
    )


def create_or_load_index(document_path: str, config: dict):
    """
    Create a new Property Graph Index from a document.
    The index is persisted in Neo4j, not locally.
    """
    # Create components
    graph_store = create_graph_store(config)
    embed_model = create_embed_model(config)
    llm = create_llm(config)

    # Set up LlamaIndex global settings
    Settings.embed_model = embed_model
    Settings.llm = llm

    # Load documents
    if os.path.isfile(document_path):
        # Single file
        documents = SimpleDirectoryReader(
            input_files=[document_path],
            file_extractor={
                # ".pdf": SmartPDFLoader(llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all")
                ".pdf": PDFReader(return_full_document=True)
            }
        ).load_data()
    elif os.path.isdir(document_path):
        # Directory of files
        documents = SimpleDirectoryReader(
            document_path,
            file_extractor = {
                # ".pdf": SmartPDFLoader(llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all")
                ".pdf": PDFReader(return_full_document=True)
            }
        ).load_data()
    else:
        raise ValueError(f"Path {document_path} is neither a file nor a directory")

    # Create ingestion pipeline with custom transformer
    print(f"Creating ingestion pipeline from {document_path}...")

    # Run documents through pipeline
    print("Parsing documents and building property graph...")

    # Build the index from transformed nodes
    index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        embed_model=embed_model,
        llm=llm,
        transformations=[
            TitleExtractor(),
            WoodcrestHillsDocumentParser(llm=llm, embed_model=embed_model),
            QuestionsAnsweredExtractor(),
        ],
        use_async=False,
    )
    graph_store.close()

    print("Index created and stored in Neo4j.")


def query_mode(index, config):
    """Interactive query mode."""
    llm = create_llm(config)

    query_engine = index.as_query_engine(
        llm=llm,
        include_text=True,
        similarity_top_k=2,
    )

    print("\n" + "=" * 60)
    print("Property Graph Index Query Engine")
    print("=" * 60)
    print("Ask questions about your document.")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            query = input("Question: ").strip()
            if query.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if not query:
                continue

            response = query_engine.query(query)
            print(f"\nAnswer: {response.response}\n")

            # Show source nodes if available
            if hasattr(response, "source_nodes") and response.source_nodes:
                print("Sources:")
                for i, node in enumerate(response.source_nodes, 1):
                    print(f"  {i}. {node.text[:200]}...")

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")


def single_query(index, query: str, config):
    """Single query mode."""
    llm = create_llm(config)

    query_engine = index.as_query_engine(
        llm=llm,
        include_text=True,
        similarity_top_k=2,
    )

    response = query_engine.query(query)
    print(f"Answer: {response.response}")

    if hasattr(response, "source_nodes") and response.source_nodes:
        print("\nSources:")
        for i, node in enumerate(response.source_nodes, 1):
            print(f"  {i}. {node.text[:200]}...")


def get_config():
    """Get configuration from environment variables."""
    required = {
        "NEO4J_USERNAME": None,
        "NEO4J_PASSWORD": None,
        "NEO4J_URL": None,
        "EMBED_MODEL": None,
        "EMBED_BASE_URL": None,
        "LLM_MODEL": None,
        "LLM_BASE_URL": None,
    }

    config = {}
    for key in required.keys():
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        config[key] = value

    return config


def main():
    if len(sys.argv) < 2:
        print("Usage: python index_documents.py <document_path> [query]")
        print("\nIf no query is provided, enters interactive chat mode.")
        sys.exit(1)

    document_path = sys.argv[1]

    if not os.path.exists(document_path):
        print(f"Error: Path '{document_path}' does not exist.")
        sys.exit(1)

    # Get configuration
    try:
        config = get_config()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set the required environment variables in your .env file.")
        sys.exit(1)

    create_or_load_index(document_path, config)


if __name__ == "__main__":
    main()
