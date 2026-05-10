import lancedb
from langchain_openai import OpenAIEmbeddings
from .base import ContextProvider

_OVERVIEW_ROWS = 10  # get_overview 直接取前 N 行，不做向量搜索


class RAGProvider(ContextProvider):
    def __init__(self, db_path: str, top_k: int = 5) -> None:
        self._db_path = db_path
        self._top_k = top_k
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._db = lancedb.connect(db_path)
        self._table = self._db.open_table("chunks")

    def get_overview(self) -> list[str]:
        # 直接扫表取前 N 条，不做向量搜索，避免无意义 query 污染结果
        rows = self._table.to_pandas()["text"].tolist()
        return rows[:_OVERVIEW_ROWS]

    def retrieve(self, query: str) -> list[str]:
        vector = self._embeddings.embed_query(query)
        results = self._table.search(vector).limit(self._top_k).to_list()
        return [r["text"] for r in results]
