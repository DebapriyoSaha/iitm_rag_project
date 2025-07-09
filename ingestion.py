from dotenv import load_dotenv
from langchain.text_splitter import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import re

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# urls = [
#     "https://docs.google.com/document/d/e/2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjsGrTP5ShhZK8N6ZSPX89lexKx86QPAUswSzGLsOA/pub",
#     "https://docs.google.com/document/d/e/2PACX-1vRKOWaLjxsts3qAM4h00EDvlB-GYRSPqqVXTfq3nGWFQBx91roxcU1qGv2ksS7jT4EQPNo8Rmr2zaE9/pub?urp=gmail_link#h.cbcq4ial1xkk",
#     "https://study.iitm.ac.in/ds/",
#     "https://study.iitm.ac.in/ds/academics.html",
#     "https://study.iitm.ac.in/ds/admissions.html",
#     "https://study.iitm.ac.in/ds/aboutIITM.html",
#     "https://study.iitm.ac.in/ds/academic_calendar.html",
# ]

# # 2. Load documents
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# # 3. Acronym Extraction and Expansion


# def extract_acronyms(text):
#     pattern = r"([A-Z][A-Za-z\s]+)\s+\(([A-Z]{2,})\)"
#     matches = re.findall(pattern, text)
#     return {acr: full.strip() for full, acr in matches}

# def expand_acronyms(text, acronym_map):
#     for acr, full in acronym_map.items():
#         pattern = r"\b" + re.escape(acr) + r"\b(?!\s*\()"
#         text = re.sub(pattern, f"{acr} ({full})", text)
#     return text

# # Apply preprocessing
# preprocessed_docs = []
# for doc in docs_list:
#     acronyms = extract_acronyms(doc.page_content)
#     expanded = expand_acronyms(doc.page_content, acronyms)
#     doc.page_content = expanded
#     preprocessed_docs.append(doc)

# # 4. First pass: HTML-aware header-based splitting
# html_splitter = HTMLHeaderTextSplitter(
#     headers_to_split_on=[
#         ("h1", "Header 1"),
#         ("h2", "Header 2"),
#         ("h3", "Header 3"),
#         ("h4", "Header 4"),
#     ]
# )

# structured_chunks = []
# for doc in preprocessed_docs:
#     chunks = html_splitter.split_text(doc.page_content)
#     for chunk in chunks:
#         # Copy metadata from original doc to each chunk
#         chunk.metadata.update(doc.metadata)
#         structured_chunks.append(chunk)

# # 5. Second pass: Recursive chunking by size
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=300,
#     chunk_overlap=50,
# )

# final_chunks = text_splitter.split_documents(structured_chunks)

# # 6. Optional: Add source metadata
# for chunk in final_chunks:
#    chunk.metadata["source"] = chunk.metadata.get("source", "IITM DS Website")
#    chunk.metadata["url"] = chunk.metadata.get("url", chunk.metadata["source"])


# # Create glossary chunks
# acronym_chunks = [
#     Document(page_content=f"{acr} stands for {definition}, a course in the BS Degree", metadata={"source": "Acronym Glossary"})
#     for acr, definition in {
#         "MLT": "Machine Learning Techniques",
#         "MLF": "Machine Learning Foundations",
#         "MLP": "Machine Learning Practice",
#         "BDM": "Business Data Management",
#         "PDSA": "Programming Data Structures and Algorithms using Python",
#         "BA": "Business Analytics",
#         "TDS": "Tools in Data Science",
#         "MAD" : "Mordern Application Development",
#         "AppDev": "Application Development",
#         "ST": "Software Testing",
#         "DSA": "Data Structures and Algorithms",
#         "AI": "Artificial Intelligence",
#         "DS": "Data Science",
#         "CV": "Computer Vision",
#         "NLP": "Natural Language Processing",
#         "LLM": "Large Language Models",
#         "MLOPS": "Machine Learning Operations",
#         "DBMS": "Database Management Systems",
#         "ADS": "Algorithms for Data Science",
#         "Gen AI": "Generative AI",
#     }.items()
# ]

# # Merge everything
# all_chunks = final_chunks + acronym_chunks


# # 9. Create vector store
# vectorstore = Chroma.from_documents(
#     documents=all_chunks,
#     collection_name="rag-chroma",
#     embedding=embeddings,
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()
