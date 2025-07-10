import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

GROQ_KEY = os.getenv("GROQ_API_KEY")


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: bool = Field(
        description="Return 'true' if the document is relevant to the question, 'false' otherwise."
    )


def get_retrieval_grader(model_name: str) -> RunnableSequence:
    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name=model_name)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    Your response must be a structured function call to `GradeDocuments` with a single boolean field: `binary_score`.\n
    - Return `true` if the document is relevant to the question. \n
    - Return `false` if the document is not relevant to the question. \n
    Never return text or explain your reasoning. Only call the function."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    return grade_prompt | structured_llm_grader
