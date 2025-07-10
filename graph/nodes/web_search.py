from typing import Any, Dict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()
# web_search_tool = TavilySearch(max_results=3)


# def web_search(state: GraphState) -> Dict[str, Any]:
#     print("---WEB SEARCH---")
#     question = state["question"]
#     existing_documents = state.get("documents", [])
#     new_documents = []

#     tavily_results = web_search_tool.invoke({"query": question})["results"]
#     for result in tavily_results:
#         content = result.get("content", "")
#         url = result.get("url", "")
#         title = result.get("title", "Web Search Result")

#         doc = Document(
#             page_content=content, metadata={"source": title, "url": url, "title": title}
#         )
#         new_documents.append(doc)

#     combined_documents = existing_documents + new_documents
#     return {"documents": combined_documents, "question": question}


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---DUCKDUCKGO SEARCH---")
    question = state["question"]
    existing_documents = state.get("documents", []) or []
    new_documents = []

    with DDGS() as ddgs:
        results = ddgs.text(question, max_results=3)
        for result in results:
            content = result.get("body", "")
            url = result.get("href", "")
            title = result.get("title", "DuckDuckGo Search Result")

            doc = Document(
                page_content=content,
                metadata={"source": title, "url": url, "title": title},
            )
            new_documents.append(doc)

    combined_documents = existing_documents + new_documents
    return {"documents": combined_documents, "question": question}

if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
