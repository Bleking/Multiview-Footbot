from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "rules" / "laws-of-the-game-202627-korean.pdf"

def load_documents(file_path=FILE_PATH):
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    return docs
