from .base import ContextProvider
from .direct import DirectProvider
from .rag import RAGProvider
from ..indexing.pipeline import estimate_tokens

AUTO_THRESHOLD_TOKENS = 100_000

def create_provider(source_path: str, strategy: str, db_path: str) -> ContextProvider:
    if strategy == "direct":
        return DirectProvider(source_path=source_path)
    if strategy == "rag":
        return RAGProvider(db_path=db_path)
    # strategy == "auto"
    token_count = estimate_tokens(source_path)
    if token_count <= AUTO_THRESHOLD_TOKENS:
        return DirectProvider(source_path=source_path)
    return RAGProvider(db_path=db_path)
