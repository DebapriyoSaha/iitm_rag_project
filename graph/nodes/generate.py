from typing import Any, Dict
from graph.chains.generation import get_generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    model_name = state.get("selected_model", "gpt-4")  # default fallback

    generation_chain = get_generation_chain(model_name)
    generation = generation_chain.invoke({"context": documents, "question": question})

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }
