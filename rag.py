
from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.9,
            max_tokens=500
        )

    if vector_store is None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),
        )


def process_urls(urls):
    """Load website content, split it into chunks, and store it in Chroma."""

    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store..."
    vector_store.reset_collection()

    yield "Loading website data..."
    loader = WebBaseLoader(web_paths=urls)
    data = loader.load()

    if not data:
        raise ValueError("No documents were loaded from the supplied URLs.")

    yield "Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
    )
    docs = text_splitter.split_documents(data)

    if not docs:
        raise ValueError("No document chunks were created.")

    yield "Adding chunks to the vector database..."
    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(documents=docs, ids=uuids)

    yield f"Finished storing {len(docs)} chunks."


def generate_answer(query):
    """Retrieve relevant website content and generate an answer."""

    initialize_components()

    if vector_store is None:
        raise RuntimeError("Vector database is not initialized.")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
    )

    result = chain.invoke({"question": query})

    answer = result.get("answer", "")
    sources = result.get("sources", "")

    return answer, sources


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
    ]

    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer(
        "What was the 30-year fixed mortgage rate, and on what date?"
    )

    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
