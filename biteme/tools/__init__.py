from .filesystem import read_file, write_file, search_files_by_name, search_files_by_content, file_outline
from .github import github_list_tree, github_read_file, github_search_code
from .web import tavily_search

ALL_TOOLS = [
    github_list_tree,
    github_read_file,
    github_search_code,
    read_file,
    file_outline,
    write_file,
    search_files_by_name,
    search_files_by_content,
    tavily_search,
]

READONLY_TOOLS = [t for t in ALL_TOOLS if t.name != "write_file"]
