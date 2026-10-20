import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

def ingest_pdf():
    doc = PyPDFLoader(str(PDF_PATH)).load()
    splits = chunk_file(doc)
    enriched_doc = [Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
        for d in splits]
    ids = [f"doc-{i}" for i in range(len(enriched_doc))]

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )

    store.add_documents(documents=enriched_doc, ids=ids)
    pass


def check_environment():
    for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set")


def chunk_file(doc: Document):
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, add_start_index=False).split_documents(doc)
    if not splits:
        raise SystemExit(0)
    else:
        return splits

if __name__ == "__main__":
    ingest_pdf()