import operator
import json
import asyncio
import pandas as pd
from ollama import AsyncClient
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

# =====================================================
# CATALOG WORKER STATE
# =====================================================
class CatalogState(TypedDict):
    query: str
    context: Annotated[list[str], operator.add]
    answer: str
    attempts: int

# =====================================================
# QUERY EXTRACTION NODE
# =====================================================
async def catalog_query_node(state: CatalogState) -> dict:
    print("[Catalog Worker] Extracting search filters...")
    system_prompt = (
        "You are a bookstore assistant.\n"
        "Extract only these fields if present:\n"
        "- title\n"
        "- author\n"
        "- genre\n\n"
        "Return ONLY valid JSON."
    )
    response = await AsyncClient().chat(
        model="llama3",
        format="json",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]}
        ]
    )
    return {
        "answer": response["message"]["content"].strip()
    }

# =====================================================
# CSV SEARCH NODE
# =====================================================
async def catalog_search_node(state: CatalogState) -> dict:
    print("[Catalog Worker] Searching books.csv...")
    try:
        filters = json.loads(state["answer"])
    except:
        return {"answer": "Unable to understand the search query."}
    try:
        df = pd.read_csv("books.csv")
        result = df
        if "title" in filters and filters["title"]:
            result = result[
                result["title"].str.contains(
                    filters["title"],
                    case=False,
                    na=False
                )
            ]
        if "author" in filters and filters["author"]:
            result = result[
                result["author"].str.contains(
                    filters["author"],
                    case=False,
                    na=False
                )
            ]
        if "genre" in filters and filters["genre"]:
            result = result[
                result["genre"].str.contains(
                    filters["genre"],
                    case=False,
                    na=False
                )
            ]
        if result.empty:
            return {
                "answer": "No matching books found."
            }
        output = []
        for _, row in result.iterrows():
            stock = "In Stock" if row["stock"] > 0 else "Out of Stock"
            output.append(
                f"Title : {row['title']}\n"
                f"Author : {row['author']}\n"
                f"Genre : {row['genre']}\n"
                f"Price : ₹{row['price']}\n"
                f"Stock : {stock} ({row['stock']} copies)\n"
            )
        return {
            "answer": "\n-----------------\n".join(output)
        }
    except FileNotFoundError:
        return {
            "answer": "books.csv not found."
        }
    except Exception as e:
        return {
            "answer": f"Database Error : {str(e)}"
        }

# =====================================================
# BUILD CATALOG GRAPH
# =====================================================
catalog_graph = StateGraph(CatalogState)
catalog_graph.add_node("query", catalog_query_node)
catalog_graph.add_node("search", catalog_search_node)
catalog_graph.set_entry_point("query")
catalog_graph.add_edge("query", "search")
catalog_graph.add_edge("search", END)
catalog_worker = catalog_graph.compile()

# =====================================================
# RECOMMENDATION WORKER STATE
# =====================================================
class RecommendationState(TypedDict):
    query: str
    answer: str
    attempts: int
    is_valid:bool

# =====================================================
# RECOMMENDATION NODE
# =====================================================
async def recommendation_node(state: RecommendationState) -> dict:
    print("[Recommendation Worker] Generating recommendations...")
    system_prompt = (
        "You are a bookstore recommendation assistant.\n"
        "Recommend only books from the inventory provided by the user.\n"
        "do not invent any book names.\n"
        "if a book has stock 0 do not recommend it"
    )
    response = await AsyncClient().chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": state["query"]
            }
        ]
    )
    return {
        "answer": response["message"]["content"],
        "attempts":state.get("attempts",0)+1
    }

#============================================
#Recommendation verify node
#============================================
async def recommendation_verify_node(state: RecommendationState)->dict:
    df = pd.read_csv("books.csv")
    answer = state["answer"].lower()
    for _, row in df.iterrows():
        if row["title"].lower() in answer:
            if row["stock"] > 0:
                print("[Recommendation Worker] Recommendation verified.")
                return {"is_valid": True}
            else:
                print("[Recommendation Worker] Book out of stock. Retrying...")
                return {"is_valid": False}
    print("[Recommendation Worker] Recommended book not found. Retrying...")
    return {"is_valid": False}

#=============================================
#Recommendation state
#=============================================
async def recommendation_state(state: RecommendationState)->dict:
    if state["is_valid"]:
        return "done"
    if state["attempts"] >= 3:
        return "done"
    return "retry"

# =====================================================
# BUILD RECOMMENDATION GRAPH
# =====================================================
recommendation_graph = StateGraph(RecommendationState)
recommendation_graph.add_node("recommend",recommendation_node)
recommendation_graph.set_entry_point("recommend")
recommendation_graph.add_node("verify",recommendation_verify_node)
recommendation_graph.add_edge("recommend","verify")
recommendation_graph.add_conditional_edges("verify",recommendation_state,{
        "retry": "recommend",
        "done": END
        })
recommendation_worker = recommendation_graph.compile()

# =====================================================
# SUPERVISOR STATE
# =====================================================
class SupervisorState(TypedDict):
    query: str
    catalog_result: str
    recommendation_result: str
    final_response: str

# =====================================================
# ORCHESTRATOR NODE
# =====================================================
async def orchestrator_node(state: SupervisorState) -> dict:
    print("[Supervisor] Understanding customer request...")
    system_prompt = (
        "You are a routing assistant for a bookstore.\n"
        "Split the user query into two independent tasks.\n"
        "Return ONLY JSON with keys:\n"
        "catalog_query\n"
        "recommendation_query\n"
        "If one task is unnecessary, return an empty string."
        "Never generate SQL, URLs, code, or API calls. Return plain English tasks only."
    )
    response = await AsyncClient().chat(
        model="llama3",
        format="json",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": state["query"]
            }
        ]
    )
    try:
        queries = json.loads(response["message"]["content"])
        catalog_query = queries.get("catalog_query", "")
        recommendation_query = queries.get("recommendation_query", "")
    except:
        catalog_query = state["query"]
        recommendation_query = state["query"]
    print("Catalog Task :", catalog_query)
    print("Recommendation Task :", recommendation_query)
    tasks = {}
    if catalog_query:
        tasks["catalog"] = catalog_worker.ainvoke({
            "query": catalog_query,
            "context": [],
            "answer": "",
            "attempts": 0
        })
    if recommendation_query:
        tasks["recommendation"] = recommendation_worker.ainvoke({
            "query": recommendation_query,
            "answer": "",
            "attempts":0,
            "is_valid":False
        })
    catalog_result = "No catalog search requested."
    recommendation_result = "No recommendation requested."
    if tasks:
        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )
        completed = dict(zip(tasks.keys(), results))
        if "catalog" in completed:
            result = completed["catalog"]
            if isinstance(result, Exception):
                catalog_result = f"Catalog Worker Failed : {result}"
            else:
                catalog_result = result["answer"]
        if "recommendation" in completed:
            result = completed["recommendation"]
            if isinstance(result, Exception):
                recommendation_result = (
                    f"Recommendation Worker Failed : {result}"
                )
            else:
                recommendation_result = result["answer"]
    return {
        "catalog_result": catalog_result,
        "recommendation_result": recommendation_result
    }

# =====================================================
# FINAL RESPONSE NODE
# =====================================================
async def final_response_node(state: SupervisorState) -> dict:
    print("\n[Supervisor] Preparing final response...")
    system_prompt = (
    "You are a bookstore assistant.\n"
    "Use ONLY the books present in Catalog Result.\n"
    "Never invent book names.\n"
    "Never recommend books that are not present in Catalog Result.\n"
    "If a book is out of stock, recommend another IN-STOCK book from Catalog Result.\n"
    "If no programming books are in stock, clearly say so.\n"
)
    user_prompt = (
        f"User Query:\n{state['query']}\n\n"
        f"Catalog Result:\n{state['catalog_result']}\n\n"
        f"Recommendation Result:\n{state['recommendation_result']}"
    )
    response = await AsyncClient().chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    return {
        "final_response": response["message"]["content"]
    }

# =====================================================
# BUILD MAIN GRAPH
# =====================================================
main_graph = StateGraph(SupervisorState)
main_graph.add_node("orchestrator",orchestrator_node)
main_graph.add_node("final_response",final_response_node)
main_graph.set_entry_point( "orchestrator")
main_graph.add_edge( "orchestrator", "final_response")
main_graph.add_edge("final_response",END)
bookstore_app = main_graph.compile()

# =====================================================
# CHAT LOOP
# =====================================================
async def chat():
    print("Type 'exit' to stop.\n")
    while True:
        user = input("\nCustomer : ")
        if user.lower() == "exit":
            break
        result = await bookstore_app.ainvoke({
            "query": user,
            "catalog_result": "",
            "recommendation_result": "",
            "final_response": ""
        })
        print("\nAssistant:\n")
        print(result["final_response"])
if __name__ == "__main__":
    asyncio.run(chat())
