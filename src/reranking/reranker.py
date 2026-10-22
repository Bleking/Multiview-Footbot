from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def get_reranker(retriever, top_n=3):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")  # 다국어 reranker이고, query와 passage를 함께 넣어서 relevance score를 직접 계산하는 cross-encoder 방식
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    return compression_retriever
