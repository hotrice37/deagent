"""
vector_db_manager.py
Manages interactions with the Pinecone vector database for storing and querying embeddings.
"""

# General Imports
import time
import uuid
from typing import List, Optional

# Pinecone Imports - Using the new Pinecone client
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException # Specific exception from pinecone.exceptions

# LangChain Imports - Ollama for embeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

class VectorDBManager:
    """Manages the interaction with the Pinecone serverless vector database."""
    def __init__(
            self,
            index_name: str,
            cloud: str,
            region: str,
            api_key: str,
            embedding_model_name: str
        ):
        """
        Initializes the VectorDBManager with Pinecone.
        :param index_name: The name of the Pinecone index.
        :param cloud: The cloud provider for the serverless index (e.g., 'aws', 'gcp').
        :param region: The specific region for the serverless index (e.g., 'us-east-1').
        :param api_key: Your Pinecone API key.
        :param embedding_model_name: The Ollama model name for embeddings.
        """
        self.embeddings_model = OllamaEmbeddings(model=embedding_model_name)
        self.pinecone = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.cloud = cloud
        self.region = region

        # Determine embedding dimension dynamically
        try:
            temp_ollama_embeddings = OllamaEmbeddings(model=embedding_model_name)
            sample_embedding = temp_ollama_embeddings.embed_query("test")
            self.embedding_dimension = len(sample_embedding)
            print(f"Embedding dimension detected for {index_name}: {self.embedding_dimension}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to get embedding dimension from Ollama: {e}. "
                f"Ensure Ollama is running and model '{embedding_model_name}' is pulled."
            ) from e
        index_names = [index["name"] for index in self.pinecone.list_indexes().get("indexes", [])]

        print(f"DEBUG_VDB: Available Pinecone indexes: {index_names}")
        # Index creation logic with try-except for graceful handling
        if index_name not in index_names:
            print(f"Attempting to create Pinecone serverless index '{index_name}' in {cloud} {region}...")
            try:
                self.pinecone.create_index(
                    name=index_name,
                    dimension=self.embedding_dimension,
                    metric='cosine',
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
                print(f"Pinecone serverless index '{index_name}' created successfully.")
            except PineconeApiException as e:
                # Check for "already exists" message or 409 status code
                if e.status == 409 or "already exists" in str(e).lower() or "ALREADY_EXISTS" in str(e):
                    print(f"Pinecone index '{index_name}' already exists (caught during creation attempt).")
                else:
                    raise e # Re-raise if it's another type of error
            except Exception as e:
                print(f"An unexpected error occurred during index creation: {e}")
                raise e
        else:
            print(f"Pinecone index '{index_name}' already exists, skipping creation.")

        # Always wait for the index to be ready, whether newly created or pre-existing
        print(f"Waiting for index '{index_name}' to be ready...")
        # Add a timeout for robustness in production
        timeout_seconds = 300 # 5 minutes
        start_time = time.time()
        while not self.pinecone.describe_index(index_name).status['ready']:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Pinecone index '{index_name}' did not become ready within {timeout_seconds} seconds.")
            time.sleep(1)
        print(f"Index '{index_name}' is ready.")

        self.index = self.pinecone.Index(index_name)

    def add_documents_batch(self, texts: List[str], metadatas: List[dict] = None, doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        Adds a batch of documents to the Pinecone index.
        :param texts: The list of text contents to embed and upsert.
        :param metadatas: A list of metadata dictionaries, corresponding to `texts`.
        :param doc_ids: Optional list of specific IDs to use for the documents. If provided, must match length of texts.
                        If not provided, UUIDs will be generated.
        :return: A list of IDs of the upserted vectors.
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)

        if doc_ids and len(doc_ids) != len(texts):
            raise ValueError("If 'doc_ids' are provided, their length must match the length of 'texts'.")

        embeddings = self.embeddings_model.embed_documents(texts)

        vectors_to_upsert = []
        ids = []
        for i, (text_content, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            # Use provided ID if available, otherwise generate UUID
            current_id = doc_ids[i] if doc_ids else str(uuid.uuid4())
            ids.append(current_id)
            combined_metadata = {"text_content": text_content, **metadata}
            vectors_to_upsert.append((current_id, embedding, combined_metadata))

        # Upsert vectors to Pinecone in batches to handle large datasets efficiently
        batch_size = 100 # Adjust batch size based on Pinecone limits and network performance
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        print(f"Upserted {len(vectors_to_upsert)} documents to Pinecone index '{self.index_name}'.")
        return ids

    def query_similar_documents(self, query_text: str, k: int = 3) -> List[Document]:
        """
        Queries the Pinecone index for documents semantically similar to the query text.
        :param query_text: The text to query for.
        :param k: The number of top similar documents to retrieve.
        :return: A list of LangChain Document objects, each containing page_content and metadata.
        """
        print(f"DEBUG_VDB: Querying Pinecone index '{self.index_name}' for relevant context for: '{query_text}'...")
        # Add debug print before embedding call
        print(f"DEBUG_VDB: Generating embedding for query text from '{self.index_name}'...")
        query_embedding = self.embeddings_model.embed_query(query_text)
        print(f"DEBUG_VDB: Embedding generated for query text from '{self.index_name}'.")

        response = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        # Convert Pinecone response to LangChain Document format
        results = []
        for match in response.matches:
            page_content = match.metadata.get("text_content", "")
            metadata = {k: v for k, v in match.metadata.items() if k != "text_content"}
            results.append(Document(page_content=page_content, metadata=metadata))

        print(f"DEBUG_VDB: Found {len(results)} similar documents in index '{self.index_name}'.")
        return results
