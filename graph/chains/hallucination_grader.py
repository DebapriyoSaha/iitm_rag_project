import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# from langchain_huggingface import HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceTB/SmolLM3-3B",
#     max_new_tokens=512,
#     temperature=0.0,
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
# )

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
You are a strict grader assessing whether an LLM-generated answer is grounded in a given set of facts.\n
Your response must be a structured function call to `GradeHallucinations` with a single boolean field: `binary_score`.\n
- Return `true` if the generation is factually supported by the provided facts.\n
- Return `false` if it contains hallucinations, makes unsupported claims, or diverges from the facts.\n
Never return text or explain your reasoning. Only call the function.
"""

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Set of facts:\n\n{documents}\n\nLLM generation:\n\n{generation}",
            ),
        ]
    )

    return hallucination_prompt | structured_llm_grader
