from langchain_openai import OpenAIEmbeddings

def get_embeddings(model="text-embedding-3-small"):
    return OpenAIEmbeddings(model=model)
