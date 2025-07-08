from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
import os

GROQ_KEY = os.getenv("GROQ_API_KEY")

class GradeAnswer(BaseModel):
    binary_score: bool = Field(description="Answer addresses the question, 'true' or 'false'")

def get_answer_grader(model_name: str) -> RunnableSequence:
    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name=model_name)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing whether an answer addresses / resolves a question.
    Your response must be a structured function call to `GradeAnswer` with a single boolean field: `binary_score`.

    - Return `true` if the answer resolves the query.
    - Return `false` if the answer does not resolve the query.

        Never return text or explain your reasoning. Only call the function."""),
        ("human", "User question:\n\n{question}\n\nLLM generation: {generation}")
    ])

    return answer_prompt | structured_llm_grader
