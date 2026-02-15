#!/usr/bin/env python3
"""
Streamlit Web Interface for Property Graph Index Query

Usage:
    streamlit run streamlit_app.py

Configure via .env file (see .env.example for examples).
"""

import os
import tempfile
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Apply nest_asyncio to handle nested event loops in Streamlit
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama

# Load environment variables
load_dotenv()


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


def create_graph_store(config):
    """Create a Neo4j property graph store."""
    return Neo4jPropertyGraphStore(
        username=config["NEO4J_USERNAME"],
        password=config["NEO4J_PASSWORD"],
        url=config["NEO4J_URL"],
    )


def create_llm(config):
    """Create an Ollama LLM instance."""
    return Ollama(
        model=config["LLM_MODEL"],
        request_timeout=240.0,
        context_window=9216,
        base_url=config["LLM_BASE_URL"],
    )


def create_embed_model(config):
    """Create an Ollama embedding model instance."""
    return OllamaEmbedding(
        model_name=config["EMBED_MODEL"],
        request_timeout=240.0,
        base_url=config["EMBED_BASE_URL"],
    )


def create_index_from_file(file_path, config, progress_callback=None):
    """Create a new index from a document file."""
    if progress_callback:
        progress_callback("Loading document...")

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    if progress_callback:
        progress_callback("Creating graph store...")

    graph_store = create_graph_store(config)
    embed_model = create_embed_model(config)
    llm = create_llm(config)

    if progress_callback:
        progress_callback("Building property graph index...")

    # Set up LlamaIndex global settings
    Settings.embed_model = embed_model
    Settings.llm = llm

    index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        embed_model=embed_model,
        llm=llm,
        use_async=False,
    )

    if progress_callback:
        progress_callback("Saving index...")

    return index


def load_existing_index(config):
    """Load an existing index from Neo4j."""
    try:
        graph_store = create_graph_store(config)
        embed_model = create_embed_model(config)
        llm = create_llm(config)

        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_model=embed_model,
            llm=llm,
        )
        return index
    except Exception as e:
        st.error(f"Error loading existing index: {e}")
        return None


def get_query_engine(config, index):
    """Create a query engine from the index."""
    llm = create_llm(config)
    return index.as_query_engine(
        llm=llm,
        include_text=True,
        similarity_top_k=10,
    )


def main():
    st.set_page_config(page_title="Property Graph Index Query", layout="wide")

    st.title("Property Graph Index Query")
    st.markdown("Ask questions about your documents using a property graph index.")

    # Initialize session state
    if "config" not in st.session_state:
        try:
            st.session_state.config = get_config()
        except ValueError as e:
            st.error(str(e))
            st.info("Please configure the required environment variables in your .env file.")
            return

    # Load existing index from Neo4j at startup
    if "index" not in st.session_state:
        st.session_state.index = None
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    config = st.session_state.config

    # Try to load existing index from Neo4j
    if st.session_state.index is None:
        with st.spinner("Loading existing index from Neo4j..."):
            index = load_existing_index(config)
            if index is not None:
                st.session_state.index = index
                st.session_state.query_engine = get_query_engine(config, index)
                st.success("Index loaded from Neo4j!")

    # Sidebar for document upload and index management
    with st.sidebar:
        st.header("Document Management")

        # Show index status
        if st.session_state.index:
            st.success("Index loaded!")
            if st.button("Clear Index", use_container_width=True):
                st.session_state.index = None
                st.session_state.query_engine = None
                st.session_state.messages = []
                st.rerun()
        else:
            st.warning("No index loaded")

        st.divider()
        st.header("Upload Document")

        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt", "pdf", "docx", "md"],
            disabled=st.session_state.processing
        )

        if uploaded_file is not None:
            if st.button("Index Document", use_container_width=True, disabled=st.session_state.processing):
                handle_upload(uploaded_file, config)

    # Main content area
    if st.session_state.index and st.session_state.query_engine:
        st.divider()
        st.header("Chat")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your document...", disabled=st.session_state.processing):
            handle_chat(prompt, config)

    else:
        st.info("Upload a document to begin.")


def handle_upload(uploaded_file, config):
    """Handle document upload and indexing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())

        progress_bar = st.progress(0, text="Processing document...")

        def progress_callback(step):
            progress_bar.progress(25, text=f"Processing: {step}...")

        try:
            index = create_index_from_file(str(temp_path), config, progress_callback)

            progress_bar.progress(75, text="Creating query engine...")
            query_engine = get_query_engine(config, index)

            progress_bar.progress(100, text="Done!")

            st.session_state.index = index
            st.session_state.query_engine = query_engine
            st.session_state.messages = []

            st.success(f"Document '{uploaded_file.name}' indexed successfully!")

            # Add a system message
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I've indexed '{uploaded_file.name}'. You can now ask questions about it!"
            })

        except Exception as e:
            st.error(f"Error indexing document: {e}")
        finally:
            progress_bar.empty()
            st.rerun()


def handle_chat(prompt, config):
    """Handle chat message and display response."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.query_engine.query(prompt)
                st.markdown(response.response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.response
                })

                # Show sources if available
                if hasattr(response, "source_nodes") and response.source_nodes:
                    with st.expander("Show sources"):
                        for i, node in enumerate(response.source_nodes, 1):
                            st.markdown(f"**Source {i}:** {node.text[:500]}...")

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.pop()  # Remove the user message if error


if __name__ == "__main__":
    main()
