import re

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import get_answer_grader
from graph.chains.hallucination_grader import get_hallucination_grader
from graph.chains.router import RouteQuery, get_question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def expand_acronyms(state: GraphState) -> GraphState:
    print("---EXPANDING ACRONYMS---")

    question = state["question"]
    lowered_question = question.lower().strip()

    # 1. Check if it's a "full form" question
    full_form_patterns = [
        r"\bfull form\b",
        r"\bstands for\b",
        r"\bmeaning of\b",
        r"\bexpand\b",
        r"\bshort for\b",
        r"\babbreviation\b",
        r"\bwhat does\b.*\bstand for\b",
    ]

    for pattern in full_form_patterns:
        if re.search(pattern, lowered_question):
            print("---SKIPPED: Detected full form/abbreviation question---")
            return state  # Skip expansion

    # 2. Acronym Expansion Dictionary
    acronym_map = {
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
        "SC": "System Commands"
    }

    # 3. Expand Acronyms in Question
    words = question.split()
    expanded_words = []

    for word in words:
        upper_word = word.upper()
        if upper_word in acronym_map:
            expanded = acronym_map[upper_word]
            print(f"Expanded {word} to {expanded}")
            expanded_words.append(expanded)
        else:
            expanded_words.append(word)

    state["question"] = " ".join(expanded_words)
    return state


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)
    model_name = state.get("selected_model", "llama-3.1-8b-instant")  # default fallback

    hallucination_grader = get_hallucination_grader(model_name)
    answer_grader = get_answer_grader(model_name)

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            if retry_count >= 1:
                print("---DECISION: GENERATION NOT USEFUL AND MAX RETRIES REACHED---")
                return "fallback"
            else:
                print(
                    "---DECISION: GENERATION NOT USEFUL. WILL RETRY WITH WEB SEARCH---"
                )
                return "not useful"
    else:
        if retry_count >= 1:
            print("---DECISION: MAX RETRIES REACHED FOR HALLUCINATION---")
            return "fallback"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED. WILL RETRY---")
            return "not supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")

    question = state["question"]
    model_name = state.get("selected_model", "llama-3.1-8b-instant")  # Default fallback

    # Dynamically create the router with the correct model
    question_router = get_question_router(model_name)

    # Invoke the routing LLM
    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE
    else:
        print("---UNKNOWN DATASOURCE. DEFAULTING TO VECTORSTORE---")
        return RETRIEVE


def handle_retry(state: GraphState) -> GraphState:
    retry_count = state.get("retry_count", 0)
    retry_count += 1
    state["retry_count"] = retry_count
    print(f"Updated retry count to: {retry_count}")
    return state


workflow = StateGraph(GraphState)

workflow.add_node("expand_acronyms", expand_acronyms)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# workflow.set_conditional_entry_point(
#     route_question,
#     {
#         WEBSEARCH: WEBSEARCH,
#         RETRIEVE: RETRIEVE,
#     },
# )
workflow.set_entry_point("expand_acronyms")
workflow.add_edge("expand_acronyms", RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_node("retry_handler", handle_retry)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": "retry_handler",
        "not useful": "retry_handler",
        "useful": END,
        "fallback": END,  # end gracefully
    },
)
workflow.add_edge("retry_handler", WEBSEARCH)
workflow.add_edge(WEBSEARCH, GENERATE)
# workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
