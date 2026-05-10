from abc import ABC, abstractmethod

class ContextProvider(ABC):
    @abstractmethod
    def get_overview(self) -> list[str]:
        """冷启动用：不需要 query，直接返回源内容概览片段。"""
        ...

    @abstractmethod
    def retrieve(self, query: str) -> list[str]:
        """根据 query 返回相关内容片段列表。"""
        ...
