import json
import os
import pandas as pd
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field, field_validator
from cryptography.fernet import Fernet
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import configuration
from config import ADMIN_CODE, SYSTEM_PROMPT, FILE_ENCRYPTION_KEY

# Initialize encryption
cipher_suite = Fernet(FILE_ENCRYPTION_KEY)
DATA_FILE = "customer_data.enc"

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class inventory_input(BaseModel):
    item_name: str = Field(description="The specific name of the product. Must not be empty.")
    color: str = Field(default="", description="The color of the product, if specified.")

class policy_input(BaseModel):
    search_query: str = Field(description="A detailed description of the policy being requested.")

    @field_validator('search_query')
    @classmethod
    def prevent_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty.")
        return cleaned

class store_credit_input(BaseModel):
    user_id: str = Field(description='The unique identifier of the user')
    change: int = Field(description='The amount of credits to be awarded to the user (1 credit = 1 Rs.)')
    admin_code: str = Field(description='The internal authorization code required to process this transaction.')

    @field_validator('change')
    @classmethod
    def validate_change(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Credit amount must be greater than zero.")
        return value

class order_tracking_input(BaseModel):
    order_id: str = Field(description="The unique order ID provided by the user.")

@tool(args_schema=inventory_input)
def fetch_inventory(item_name: str, color: str = "") -> str:
    """Check the inventory for an item, its stock levels, and its price.In case of any refund use this tool to find the item's value"""
    try:
        df = pd.read_csv("inventory.csv")
        matches = df[df['name'].str.contains(item_name, case=False, na=False)]
        if color:
            matches = matches[matches['color'].str.contains(color, case=False, na=False)]
        if matches.empty:
            return f"No stock found for '{item_name}'" + (f" in '{color}'." if color else ".")

        result_lines = [f"- {row['name']} ({row['color']}): {row['stock']} units in stock, Price: {row['price']} Rs." for _, row in matches.iterrows()]
        return f"Inventory results for '{item_name}':\n" + "\n".join(result_lines)
    except Exception as e:
        return f"SYSTEM ERROR: Failed to query inventory. {str(e)}"


try:
    loader = TextLoader("policy.txt", encoding="utf-8")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(raw_docs)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embedding_model)
    policy_retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={"k": 2})
    db_initialized = True
except Exception as e:
    print(f"Warning: RAG initialization failed: {e}")
    db_initialized = False

@tool(args_schema=policy_input)
def search_store_policies(search_query: str) -> str:
    """Searches the store policies document for a specific topic."""
    if not db_initialized:
        return "SYSTEM ERROR: Policy database is currently offline."
    try:
        relevant_docs = policy_retriever.invoke(search_query)
        if relevant_docs:
            result_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            return f"Relevant policy context for '{search_query}':\n\n{result_text}"
        return f"No relevant policy found for '{search_query}'."
    except Exception as err:
        return f"SYSTEM ERROR: Vector search failed. {str(err)}"

@tool(args_schema=store_credit_input)
def issue_store_credit(user_id: str, change: int, admin_code: str) -> str:
    """Issues store credit to a user's account using an encrypted local file."""
    if admin_code != ADMIN_CODE:
        return "SYSTEM ERROR: Unauthorized. Invalid admin override code."

    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "rb") as f:
                encrypted_data = f.read()
                decrypted_text = cipher_suite.decrypt(encrypted_data).decode('utf-8')
                ledger = json.loads(decrypted_text)
        else:
            ledger = {}

        current_balance = ledger.get(user_id, 0)
        ledger[user_id] = current_balance + change

        json_data = json.dumps(ledger).encode('utf-8')
        encrypted_output = cipher_suite.encrypt(json_data)

        with open(DATA_FILE, "wb") as f:
            f.write(encrypted_output)

        return f"SUCCESS: {change} credits issued to {user_id}."

    except Exception as e:
        return f"SYSTEM ERROR: Failed to process secure transaction. {str(e)}"

@tool(args_schema=order_tracking_input)
def track_order(order_id: str) -> str:
    """Track the status, items, and price of a customer's order using the order ID."""
    try:
        df = pd.read_csv("orders.csv")

        df['order_id'] = df['order_id'].astype(str)
        match = df[df['order_id'] == str(order_id)]

        if match.empty:
            return f"Order lookup failed: No order found with ID '{order_id}'."

        row = match.iloc[0]
        return (f"Order ID: {order_id}\n"
                f"Item: {row['item_name']}\n"
                f"Status: {row['status']}\n"
                f"Purchase Date: {row['purchase_date']}\n"
                f"Price: {row['price']} Rs.")
    except Exception as e:
        return f"SYSTEM ERROR: Failed to query orders database. {str(e)}"

llm = ChatOllama(model="qwen3:8b")
tools = [fetch_inventory, search_store_policies, issue_store_credit, track_order]
llm_with_tools = llm.bind_tools(tools)

def chat_node(state: AgentState) -> dict:
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    messages = [sys_msg] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

app = StateGraph(AgentState)
app.add_node("chat_node", chat_node)
app.add_node("tools", tool_node)
app.add_edge(START, "chat_node")
app.add_conditional_edges("chat_node", tools_condition)
app.add_edge("tools", "chat_node")
graph = app.compile()

async def chat_loop():
    print("\n=== TechStore Guarded Domain Agent Started ===")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
        if not user_input.strip():
            continue

        final_state = await graph.ainvoke({"messages": [HumanMessage(content=user_input)]})
        print("\nAgent:", final_state["messages"][-1].content)

asyncio.run(chat_loop())
