import os

from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

GROQ_KEY = os.getenv("GROQ_API_KEY")

def get_generation_chain(model_name: str):
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate.from_template("""
    You are a helpful AI assistant trained to answer queries about the IIT Madras BS Degree program. Use only the information from the context below to respond accurately.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
    - Utilize the context provided for accurate and specific information.
    - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
                                          
    STRICT INSTRUCTIONS:
    - Never return your own opinions or assumptions.
    - Never return your reasoning or thought process.

    Answer:
    """)

    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name=model_name,temperature=0.1)
    return prompt | llm | StrOutputParser()
