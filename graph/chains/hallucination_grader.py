from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import os

GROQ_KEY = os.getenv("GROQ_API_KEY")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(
        description="Return true if the answer is grounded in the facts, false if it is not."
    )


def get_hallucination_grader(model_name: str) -> RunnableSequence:
    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name=model_name)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system_prompt = """
You are a strict grader assessing whether an LLM-generated answer is grounded in a given set of facts.
Your response must be a structured function call to `GradeHallucinations` with a single boolean field: `binary_score`.

- Return `true` if the generation is factually supported by the provided facts.
- Return `false` if it contains hallucinations, makes unsupported claims, or diverges from the facts.

Never return text or explain your reasoning. Only call the function.
"""

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation:\n\n{generation}")
    ])

    return hallucination_prompt | structured_llm_grader
