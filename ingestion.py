from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [
    UnstructuredLoader(
        web_url=url, chunking_strategy="basic", max_characters=1000000
    ).load()
    for url in urls
]
doc_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(doc_list)

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

# Uncomment when creating vector store for first time

# vector_store = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embeddings,
#     persist_directory="./langgraph-tutorial/.chroma",
# )

retriever = Chroma(
    collection_name="rag-chroma", embedding_function=embeddings, persist_directory="./langgraph-tutorial/.chroma"
).as_retriever()
