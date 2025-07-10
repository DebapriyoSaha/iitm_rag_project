import asyncio
from typing import Any, Dict, List

from langchain_core.documents import Document

from graph.chains.retrieval_grader import get_retrieval_grader
from graph.state import GraphState


async def grade_single_doc(
    grader, question: str, document: Document
) -> tuple[bool, Document]:
    try:
        score = await grader.ainvoke(
            {"question": question, "document": document.page_content}
        )
        return score.binary_score, document
    except Exception as e:
        print(f"Grading failed for document: {e}")
        return False, document


async def async_grade_documents(
    question: str, documents: List[Document], model_name: str
):
    grader = get_retrieval_grader(model_name)
    tasks = [grade_single_doc(grader, question, doc) for doc in documents]
    return await asyncio.gather(*tasks)


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grades all retrieved documents in parallel to determine relevance to the question.
    Returns filtered documents and whether web search fallback is needed.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    model_name = state.get("selected_model", "llama-3.1-8b-instant")

    # Run async grading
    results = asyncio.run(async_grade_documents(question, documents, model_name))

    # Filter relevant documents
    filtered_docs = [doc for score, doc in results if score]
    web_search = len(filtered_docs) == 0

    print(f"✓ {len(filtered_docs)} of {len(documents)} documents marked relevant")
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


# from typing import Any, Dict

# # from graph.chains.retrieval_grader import retrieval_grader
# from graph.chains.retrieval_grader import get_retrieval_grader
# from graph.state import GraphState


# def grade_documents(state: GraphState) -> Dict[str, Any]:
#     """
#     Determines whether the retrieved documents are relevant to the question
#     If any document is not relevant, we will set a flag to run web search

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Filtered out irrelevant documents and updated web_search state
#     """

#     print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
#     question = state["question"]
#     documents = state["documents"]
#     model_name = state.get("selected_model", "llama-3.1-8b-instant")  # default fallback

#     filtered_docs = []
#     web_search = False

#     counter = 0
#     for d in documents:
#         retrieval_grader = get_retrieval_grader(model_name)
#         score = retrieval_grader.invoke(
#             {"question": question, "document": d.page_content}
#         )
#         grade = score.binary_score
#         if grade:
#             print("---GRADE: DOCUMENT RELEVANT---")
#             filtered_docs.append(d)
#         else:
#             print("---GRADE: DOCUMENT NOT RELEVANT---")
#             counter += 1
#             continue
#     if counter == len(documents):
#         web_search = True
#     return {"documents": filtered_docs, "question": question, "web_search": web_search}
