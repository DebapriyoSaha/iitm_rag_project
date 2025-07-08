from typing import Literal
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable

GROQ_KEY = os.getenv("GROQ_API_KEY")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Define system prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to IIT Madras BS Degree Course in Data Science & Application.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

# ✅ Function that returns the runnable with correct model
def get_question_router(model_name: str) -> Runnable:
    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name=model_name)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    return route_prompt | structured_llm_router
