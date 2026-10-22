from pathlib import Path
from langchain_chroma import Chroma

PERSIST_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION_NAME = "laws_of_the_game"

def get_vector_store(embeddings, chunks=None, persist_directory=PERSIST_DIR):
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    # Only embed the chunks the first time; later runs reuse the persisted collection
    if chunks and not vector_store.get(limit=1)["ids"]:
        vector_store.add_documents(chunks)

    return vector_store
