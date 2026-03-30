"""
ChromaDB Ingestion Script
=========================
Loads preprocessed_chunks.json into a persistent ChromaDB vector store
using LlamaIndex TextNodes and HuggingFace BGE embeddings.
"""

import json
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Configuration ────────────────────────────────────────────────────────────
CHUNKS_FILE = "preprocessed_chunks.json"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "market_reference"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def get_embed_model():
    """Return the HuggingFace embedding model."""
    return HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def ingest():
    """Load preprocessed chunks into a persistent ChromaDB collection."""

    # 1. Load JSON ─────────────────────────────────────────────────────────
    print(f"📂 Loading chunks from {CHUNKS_FILE} …")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   ✔ Loaded {len(chunks):,} chunks")

    # 2. ChromaDB persistent client & collection ──────────────────────────
    print(f"🗄️  Initialising ChromaDB at {CHROMA_DB_PATH} …")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    print(f"   ✔ Collection '{COLLECTION_NAME}' ready "
          f"(existing docs: {chroma_collection.count()})")

    # 3. Build LlamaIndex TextNodes ───────────────────────────────────────
    print("🔨 Converting chunks to TextNodes …")
    nodes = []
    for entry in chunks:
        node = TextNode(
            text=entry["content"],
            id_=entry["id"],
            metadata=entry["metadata"],
        )
        nodes.append(node)
    print(f"   ✔ Created {len(nodes):,} TextNodes")

    # 4. Embedding model ──────────────────────────────────────────────────
    print(f"🤖 Loading embedding model '{EMBED_MODEL_NAME}' …")
    embed_model = get_embed_model()

    # 5. Build index & persist via ChromaVectorStore ──────────────────────
    print("📥 Indexing nodes into ChromaDB (this may take a while) …")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"   ✔ Indexing complete — {chroma_collection.count():,} vectors stored")
    print(f"   ✔ Persistent data saved to {CHROMA_DB_PATH}/\n")


def query_index(query_text: str, top_k: int = 5):
    """
    Open the persisted ChromaDB store (no re-indexing) and retrieve
    the top-k most similar chunks for *query_text*.
    """
    print(f"🔍 Querying for: '{query_text}'  (top {top_k})\n")

    # Re-open persistent store
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = get_embed_model()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query_text)

    for i, node_with_score in enumerate(results, start=1):
        score = node_with_score.score
        node = node_with_score.node
        snippet = node.text[:200].replace("\n", " ")
        print(f"  [{i}] Score: {score:.4f}  |  ID: {node.node_id}")
        print(f"      Category: {node.metadata.get('Category', 'N/A')}")
        print(f"      Content:  {snippet}…\n")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingest()
    query_index("Software Engineer")
