"""
Pre-generate embeddings for ClawGPT
Run this script once to create the vector database before deploying.
"""

import os
import json
import requests
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
import shutil
import time

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables!")


def load_documents():
    """Load all markdown documents from the pdfs folder."""
    docs_path = Path(__file__).parent / "pdfs"
    documents = []

    if not docs_path.exists():
        raise FileNotFoundError(f"Documentation folder not found: {docs_path}")

    md_files = list(docs_path.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from content
            title = md_file.stem
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    title = line[2:].strip()
                    break

            documents.append({
                "content": content,
                "source": md_file.name,
                "title": title
            })
            print(f"  + Loaded: {md_file.name}")
        except Exception as e:
            print(f"  x Error loading {md_file.name}: {e}")

    return documents


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks."""
    chunks = []

    # Split by major sections first
    sections = text.split('\n## ')

    for i, section in enumerate(sections):
        if i > 0:
            section = '## ' + section

        # If section is small enough, keep it
        if len(section) <= chunk_size:
            if section.strip():
                chunks.append(section.strip())
        else:
            # Split large sections by paragraphs
            paragraphs = section.split('\n\n')
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

    return chunks


def get_embedding(text: str, retries: int = 3) -> list:
    """Get embedding using OpenRouter API."""
    for attempt in range(retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://clawgpt.streamlit.app",
                    "X-Title": "ClawGPT",
                },
                data=json.dumps({
                    "model": "google/gemini-embedding-001",
                    "input": text,
                    "encoding_format": "float"
                }),
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            elif response.status_code == 429:
                # Rate limited, wait and retry
                wait_time = 2 ** attempt
                print(f"\n   Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n   Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"\n   Request error: {e}")
            time.sleep(1)

    # Return zero vector as fallback
    return [0.0] * 768


def get_batch_embeddings(texts: list, batch_size: int = 10) -> list:
    """Get embeddings for a batch of texts."""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://clawgpt.streamlit.app",
                "X-Title": "ClawGPT",
            },
            data=json.dumps({
                "model": "google/gemini-embedding-001",
                "input": texts,
                "encoding_format": "float"
            }),
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        else:
            print(f"\n   Batch error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"\n   Batch request error: {e}")
        return None


def create_embeddings():
    """Create vector store with embeddings."""
    print("\n" + "="*50)
    print("ClawGPT Embedding Generator (OpenRouter)")
    print("="*50 + "\n")

    # Load documents
    print("[1/4] Loading documents...")
    documents = load_documents()

    if not documents:
        raise ValueError("No documents loaded!")

    print(f"\n[OK] Loaded {len(documents)} documents\n")

    # Split documents into chunks
    print("[2/4] Splitting documents into chunks...")
    chunks = []
    metadata = []

    for doc in documents:
        doc_chunks = split_text(doc["content"])
        for chunk in doc_chunks:
            chunks.append(chunk)
            metadata.append({
                "source": doc["source"],
                "title": doc["title"]
            })

    print(f"[OK] Created {len(chunks)} chunks\n")

    # Create embeddings using OpenRouter
    print("[3/4] Creating embeddings with OpenRouter (gemini-embedding-001)...")
    print("      This may take a few minutes...\n")

    embeddings = []
    batch_size = 5  # Process in small batches to avoid rate limits

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Try batch first
        batch_result = get_batch_embeddings(batch)

        if batch_result:
            embeddings.extend(batch_result)
        else:
            # Fallback to individual requests
            for chunk in batch:
                emb = get_embedding(chunk)
                embeddings.append(emb)

        print(f"      Progress: {min(i+batch_size, len(chunks))}/{len(chunks)}", end='\r')

        # Small delay between batches to avoid rate limits
        time.sleep(0.5)

    print(f"\n[OK] Created {len(embeddings)} embeddings\n")

    # Save to disk
    print("[4/4] Saving vector store...")

    store_dir = Path(__file__).parent / "vector_store"
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True)

    # Save embeddings as numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(store_dir / "embeddings.npy", embeddings_array)

    # Save chunks and metadata
    with open(store_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metadata": metadata}, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Vector store created successfully at: {store_dir}")
    print(f"     Total documents: {len(documents)}")
    print(f"     Total chunks: {len(chunks)}")
    print(f"     Embedding dimension: {len(embeddings[0])}")
    print("\n[DONE] Embeddings ready! Run the app with: streamlit run app.py")


if __name__ == "__main__":
    create_embeddings()
