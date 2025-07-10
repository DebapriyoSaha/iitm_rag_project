import os
import re
import uuid
from io import StringIO
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
GROQ_KEY = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "rag-project-ingestion"
llm = ChatGroq(groq_api_key=GROQ_KEY, model_name="llama3-70b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Acronym Definitions ---
ACRONYM_MAP = {
    "MLT": "Machine Learning Techniques",
    "MLF": "Machine Learning Foundations",
    "MLP": "Machine Learning Practice",
    "BDM": "Business Data Management",
    "PDSA": "Programming Data Structures and Algorithms using Python",
    "BA": "Business Analytics",
    "TDS": "Tools in Data Science",
    "MAD": "Modern Application Development",
    "AppDev": "Application Development",
    "ST": "Software Testing",
    "DSA": "Data Structures and Algorithms",
    "AI": "Artificial Intelligence",
    "DS": "Data Science",
    "CV": "Computer Vision",
    "NLP": "Natural Language Processing",
    "LLM": "Large Language Models",
    "MLOPS": "Machine Learning Operations",
    "DBMS": "Database Management Systems",
    "ADS": "Algorithms for Data Science",
    "Gen AI": "Generative AI",
}


# --- Extract All Internal Links ---
def extract_all_internal_links(base_url: str, max_depth=2) -> list[str]:
    visited = set()
    to_visit = [(base_url, 0)]
    domain = urlparse(base_url).netloc

    while to_visit:
        current_url, depth = to_visit.pop()
        if current_url in visited or depth > max_depth:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print(f"Failed to access {current_url}: {e}")
            continue

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(current_url, href)
            parsed = urlparse(full_url)

            # Skip external links and PDFs
            if parsed.netloc == domain:
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if clean_url.endswith(".pdf"):
                    continue
                if clean_url not in visited:
                    to_visit.append((clean_url, depth + 1))

    return list(sorted(visited))


# --- Extract Text and Tables ---
def extract_text_and_tables(url: str) -> Tuple[str, List[pd.DataFrame]]:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to retrieve {url}: {e}")
        return "", []

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    tables = []
    for index, table in enumerate(soup.find_all("table")):
        try:
            df = pd.read_html(StringIO(str(table)))[0]
            tables.append(df)
        except Exception as e:
            print(f"Failed to parse table {index} on {url}: {e}")
    return text, tables


# --- Expand Acronyms ---
def expand_acronyms(text: str, acronym_map: dict) -> str:
    for acronym, full_form in acronym_map.items():
        text = re.sub(rf"\b{acronym}\b", f"{acronym} ({full_form})", text)
    return text


# --- Summarize Tables ---
def summarize_table(df: pd.DataFrame, url: str) -> str:
    prompt = f"""You are an assistant who reads tables from webpages and converts them into a structured paragraph.

Here is a table from the website {url}:

{df.to_markdown(index=False)}

Write a clear and concise summary of the table capturing all the relevant information.
Make sure to include the meaning title of the table, and any important details."""
    try:
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception as e:
        print(f"LLM failed on table: {e}")
        return ""


# --- Convert Content to Documents ---
def create_documents_from_text_and_tables(
    text: str, tables: List[pd.DataFrame], url: str
) -> List[Document]:
    documents = []

    # Main text document
    expanded_text = expand_acronyms(text, ACRONYM_MAP)
    if expanded_text:
        documents.append(Document(page_content=expanded_text, metadata={"source": url}))

    # Each table summary as its own document
    for df in tables:
        summary = summarize_table(df, url)
        expanded_summary = expand_acronyms(summary, ACRONYM_MAP)
        if expanded_summary:
            documents.append(
                Document(page_content=expanded_summary, metadata={"source": url})
            )
    return documents


# --- Chunk Documents ---
def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    return splitter.split_documents(documents)


# --- Ingest to Chroma ---
def ingest_to_chroma(documents: List[Document]):
    if not documents:
        print("⚠️ No documents to ingest. Skipping.")
        return

    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="rag-chroma",
        embedding=embeddings,
        persist_directory="./.chroma",
    )
    print("✓ Ingested into Chroma.")


# --- Ingest Acronyms ---
def ingest_acronym_definitions(acronym_map: dict):
    glossary_text = "\n".join(f"{k}: {v}" for k, v in acronym_map.items())
    doc = Document(page_content=glossary_text, metadata={"source": "Acronym Glossary"})
    chunks = chunk_documents([doc])
    ingest_to_chroma(chunks)
    print("✓ Acronym glossary ingested")


# --- Main Processing Pipeline ---
def process_urls(url_list: List[str]):
    for url in url_list:
        print(f"\n--- Processing {url} ---")
        text, tables = extract_text_and_tables(url)

        if not text and not tables:
            print(f"⚠️ Skipping {url} (no retrievable content).")
            continue

        docs = create_documents_from_text_and_tables(text, tables, url)
        if not docs:
            print(f"⚠️ Skipping {url} (no valid documents).")
            continue

        chunked_docs = chunk_documents(docs)
        ingest_to_chroma(chunked_docs)
        print(f"✓ Finished processing: {url}")


# --- Entry Point ---
if __name__ == "__main__":
    urls = [
        "https://docs.google.com/document/d/e/2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjsGrTP5ShhZK8N6ZSPX89lexKx86QPAUswSzGLsOA/pub",
        "https://docs.google.com/document/d/e/2PACX-1vRKOWaLjxsts3qAM4h00EDvlB-GYRSPqqVXTfq3nGWFQBx91roxcU1qGv2ksS7jT4EQPNo8Rmr2zaE9/pub?urp=gmail_link#h.cbcq4ial1xkk",
        "https://docs.google.com/document/d/e/2PACX-1vSHXM0T-Rl2h0M9_33mEGChYIHo29UUJ0coR5YEt1_KfFaybnHlBUawBODHUwlBKqjMTc2Ie18gRRnm/pub",
    ]

    # Ingest acronyms
    ingest_acronym_definitions(ACRONYM_MAP)

    # Crawl site and ingest
    all_urls = urls + extract_all_internal_links(
        "https://study.iitm.ac.in/ds/", max_depth=11
    )
    process_urls(all_urls)
