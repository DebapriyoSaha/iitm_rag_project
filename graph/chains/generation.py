from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

GROQ_KEY = os.getenv("GROQ_API_KEY")

def get_generation_chain(model_name: str):
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name=model_name)
    return prompt | llm | StrOutputParser()
