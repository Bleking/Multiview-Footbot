from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.ingestion.loader import load_documents
from src.chunking.chunker import split_documents
from src.embeddings.embedder import get_embeddings
from src.vectordb.vector_store import get_vector_store
from src.retrieval.retriever import get_retriever
from src.reranking.reranker import get_reranker
from src.llm.llm_client import generate_answer, summarize_video

load_dotenv()

PROMPT = PromptTemplate.from_template(
    """You are a football referee assistant. Answer the question using only the
Laws of the Game excerpts below. If the excerpts do not cover it, say so.

Laws of the Game excerpts:
{context}

Question:
{question}

Answer:"""
)


def format_docs(docs):
    return "\n\n".join(
        f"[page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in docs
    )


def build_rag_chain():
    # Indexing: load -> chunk -> embed -> store
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings, chunks)

    # Retrieval + generation
    # Over-fetch candidates from Chroma, then let the cross-encoder keep the best top_n
    retriever = get_retriever(vector_store, k=20)
    reranker = get_reranker(retriever)
    llm = RunnableLambda(lambda prompt_value: generate_answer(prompt_value.to_string()))

    return (
        {"context": reranker | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def ask(chain, question, video_path=None):
    # With a clip, describe it first so the retriever can match the incident to the rules
    if video_path:
        summary = summarize_video("Describe the foul or incident in this football clip.", video_path)
        question = f"{question}\n\nVideo description:\n{summary}"

    return chain.invoke(question)


if __name__ == "__main__":
    chain = build_rag_chain()
    print(ask(chain, "What is the punishment for a reckless tackle?"))
