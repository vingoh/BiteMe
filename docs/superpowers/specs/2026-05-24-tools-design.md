# Tools 模块设计文档

**日期**: 2026-05-24  
**状态**: 待实现

---

## 背景

BiteMe 当前通过本地路径 (`source_path`) 读取代码，经 DirectProvider / RAGProvider 提供上下文。本次新增一个独立的 `tools/` 模块，提供 7 个 LangGraph 兼容的 `@tool` 装饰函数，使后续 agent 节点能够通过工具调用方式读写代码和文件。

---

## 目录结构

```
biteme/tools/
├── __init__.py          # 统一导出 ALL_TOOLS 列表
├── github.py            # 3 个 GitHub 工具
└── filesystem.py        # 4 个本地文件系统工具
```

外部使用方式：

```python
from biteme.tools import ALL_TOOLS
model.bind_tools(ALL_TOOLS)
```

---

## 工具清单

### GitHub 工具（`github.py`）

所有 GitHub 工具通过 GitHub REST API 操作，不在本地存储任何文件。

**`github_list_tree(repo_url, path="", ref="HEAD") -> str`**  
列出仓库指定路径下的目录结构，返回 JSON 字符串，每项包含 `name`、`type`（file/dir）、`path`。

**`github_read_file(repo_url, file_path, ref="HEAD") -> str`**  
读取仓库中单个文件的原始内容，返回 UTF-8 字符串。

**`github_search_code(repo_url, query, max_results=20) -> str`**  
使用 GitHub Search API 在仓库内按关键词搜索代码，返回 JSON 字符串，每项包含 `path`、`url`、匹配片段。

**参数规范**：
- `repo_url`：接受完整 URL（`https://github.com/owner/repo`）或短格式（`owner/repo`），内部统一解析为 `owner/repo`。
- `ref`：分支名、tag 或 commit SHA，默认 `"HEAD"`。
- GitHub token 从环境变量 `GITHUB_TOKEN` 读取（通过 `settings`），未设置时匿名调用（60 req/h 限制）。

**错误处理**：所有工具在 API 失败时返回包含错误描述的字符串（而非抛出异常），供 LLM 感知并决策重试或放弃。

---

### 本地文件系统工具（`filesystem.py`）

**`write_file(path, content, mode="overwrite") -> str`**  
写入本地文件。`mode` 支持 `"overwrite"`（创建或覆盖）和 `"append"`（追加）。自动创建不存在的父目录。返回成功确认或错误描述。

**`read_file(path) -> str`**  
读取本地文件内容，返回 UTF-8 字符串。文件不存在或读取失败时返回错误描述字符串。

**`search_files_by_name(directory, pattern) -> str`**  
在指定目录下按文件名模式搜索，`pattern` 支持 glob（如 `*.py`）和普通关键词（子串匹配）。返回匹配文件的相对路径列表（JSON 字符串）。

**`search_files_by_content(directory, query, file_glob="*", context_lines=2) -> str`**  
在指定目录下遍历文件，按正则/关键词搜索文件内容，返回类 grep 格式输出：

```
path/to/file.py:42: matched line
  40: context before
  43: context after
```

`file_glob` 用于限制搜索范围（如 `*.py`）。底层使用 Python 原生遍历，不依赖系统 `grep`。

---

## 配置扩展

在 `biteme/config.py` 的 `Settings` 类中新增：

```python
self.github_token: str = os.getenv("GITHUB_TOKEN", "")
```

---

## 依赖

- `requests`：GitHub API HTTP 调用（项目已有或需新增）
- 无其他新增外部依赖；本地文件操作均使用 Python 标准库

---

## 不在范围内

- GitLab / Bitbucket 支持
- 将代码 clone 到本地
- 通过 API 写回远程仓库
- 与现有 DirectProvider / RAGProvider 的自动联动

