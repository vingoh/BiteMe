from pathlib import Path
import lancedb
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

_SUPPORTED_GLOB = "**/*.{py,md,txt,ts,js,go,rs,java,yaml,toml,json}"

_SUPPORTED_EXTENSIONS = {".py", ".md", ".txt", ".ts", ".js", ".go", ".rs", ".java", ".yaml", ".toml", ".json"}


def estimate_tokens(source_path: str) -> int:
    total = 0
    path = Path(source_path)
    files = [path] if path.is_file() else list(path.rglob("*"))
    for f in files:
        if f.is_file() and f.suffix in _SUPPORTED_EXTENSIONS:
            try:
                total += len(f.read_text(errors="ignore")) // 4
            except Exception:
                pass
    return total


def build_index(source_path: str, db_path: str) -> None:
    # Python's glob does not support brace expansion, so we load each extension separately
    all_docs = []
    path = Path(source_path)
    for ext in _SUPPORTED_EXTENSIONS:
        matched = list(path.rglob(f"*{ext}"))
        for file in matched:
            try:
                loader = TextLoader(str(file), autodetect_encoding=True)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception:
                pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vectors = embeddings.embed_documents(texts)

    db = lancedb.connect(db_path)
    data = [
        {"vector": v, "text": t, "source": m.get("source", "")}
        for v, t, m in zip(vectors, texts, metadatas)
    ]
    db.create_table("chunks", data=data, mode="overwrite")
