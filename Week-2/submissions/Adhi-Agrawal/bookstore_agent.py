"""
Week 2 — Tool Calling AI Agent
Domain: BookStore Assistant

Features:
- Typed tool inputs using Pydantic
- Uses OpenLibrary API
- Uses Google Books API
- Tool class pattern
- Graceful error handling
- Ollama tool calling

Run:
    pip install requests pydantic ollama
    bookstore_agent.py
"""
import requests
import json
import ollama

from pydantic import BaseModel,Field,field_validator
from typing import Callable,Any

# ─────────────────────────────────────────────
# PYDANTIC INPUT SCHEMAS
# ─────────────────────────────────────────────

class SearchBookInput(BaseModel):
    """Input schema for searching books"""
    query: str = Field(description="Book search query")
    @field_validator("query")
    @classmethod
    def validate_query(cls, v:str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class BookDetailsInput(BaseModel):
    """input schema for fetching details of specific book."""
    title: str = Field(description="Exact book title")
    @field_validator("title")
    @classmethod
    def validate_title(cls, v:str) -> str:
        if not v.strip():
            raise ValueError( "Title cannot be empty")
        return v.strip()

class GoogleBookInput(BaseModel):
    """Input schema for Google Books lookup"""
    title: str = Field(description="Book title")
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()
    
# ─────────────────────────────────────────────
# REAL API FUNCTIONS
# OpenLibrary API
# ─────────────────────────────────────────────

def search_books(query: str) -> str:
    """search books from openlibrary using title query(No API key)"""
    try:
        response = requests.get("https://openlibrary.org/search.json",
            params={"q": query},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        docs = data.get("docs",[])[:3]
        if not docs:
            return "No books found."
        books = []
        for book in docs:
            title = book.get(
                "title",
                "Unknown Title"
            )
            books.append(title)
        return json.dumps(books,indent=2)
    except requests.RequestException as e:
        return (f"Search failed: {e}")

def get_book_details(title: str) -> str:
    """Fetch details of book from openlibrary"""
    try:
        response = requests.get("https://openlibrary.org/search.json",
            params={"title": title},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        docs = data.get( "docs", [])
        if not docs:
            return ( f"No details found for '{title}'.")
        book = docs[0]
        result = {
            "title":book.get("title","Unknown"),
            "author":(book.get("author_name",["Unknown"])[0])
        }
        return json.dumps(result,indent=2)
    except requests.RequestException as e:
        return (
            f"Could not fetch "
            f"book details: {e}"
        )
def get_google_book_info(title: str) -> str:
    try:
        response = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={"q": title,
                    "maxResults": 1
            },
            timeout=10
        )

        if response.status_code == 429:
            return get_book_details(title)
        
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            return f"No Google Books data found for '{title}'."
        volume = items[0]["volumeInfo"]
        result = {
            "title": volume.get("title", "Unknown"),
            "authors": volume.get("authors", ["Unknown"]),
            "page_count": volume.get("pageCount", "Unknown")
        }
        return json.dumps(result, indent=2)
    except requests.RequestException:
        return get_book_details(title)
# ─────────────────────────────────────────────
# TOOL CLASS
# ─────────────────────────────────────────────

class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        func: Callable[..., str],
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.func = func

    def run(self,raw_input: dict[str, Any]) -> str:
        """Validate inputs with pydantic and then call function."""
        try:
            validated = (self.input_model(**raw_input))
            return self.func(**validated.model_dump())
        except Exception as e:
            return (
                f"Tool '{self.name}' "
                f"failed: {e}"
            )

    def to_ollama_definition(self) -> dict[str, Any]:
        """Convert the tool int ollama's function from pydantic model."""
        return {
            "type": "function",
            "function": {
                "name":self.name,
                "description":self.description,
                "parameters":self.input_model.model_json_schema()
            }
        }

# ─────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="search_books",
        description=(
            "Search books using OpenLibrary ,Use this when user asks for book recommendations or searches for books."),
        input_model=SearchBookInput,
        func=search_books,
    ),

    Tool(
        name="get_book_details",
        description=(
            "Get details of a specific book author."
            ),
        input_model=BookDetailsInput,
        func=get_book_details,
    ) ,
    Tool(
    name="get_google_book_info",
    description=(
        "Get detailed book information from Google Books API including page count and categories."
        ),
    input_model=GoogleBookInput,
    func=get_google_book_info,
    ),
]

# ─────────────────────────────────────────────
# TOOL MAP
# ─────────────────────────────────────────────

TOOL_MAP: dict[str,Tool] = {
    tool.name: tool
    for tool in TOOLS
}    

# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

class BookStoreAgent:
    def __init__(self) -> None:
        self.tool_definitions = [tool.to_ollama_definition()for tool in TOOLS]
    def run(self,user_message: str) -> str:
        """process a user query using tool-calling."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a bookstore assistant. "
                    "Always use tools when book information, recommendations,or book details are required."
                    "Never explain tool calls. "
                    "Never say 'I will call a function'. "
                    "Never output JSON describing a tool call. "
                    "Call the tool directly and then answer using the tool result. "
                    "Give short,concise and readable responses."
                    "keep response under two sentence."
                )
            },

            {
                "role": "user",
                "content": user_message
            }
        ]
        print(f"\nUser: {user_message}")
        while True:
            response = ollama.chat(
                model="llama3.1",
                messages=messages,
                tools=self.tool_definitions,
            )
            msg = response["message"]
            messages.append(msg)
            if not msg.get("tool_calls"):
                return msg.get("content","")
            
            for tool_call in msg["tool_calls"]:
                func_name = (tool_call["function"]["name"])
                args = (tool_call["function"]["arguments"])
                print(f" → {func_name} ")
                f"({args})"
                if func_name in TOOL_MAP:
                    result = (TOOL_MAP[func_name].run(args))
                else:
                    result = (
                        f"Tool "
                        f"{func_name} "
                        f"not found."
                    )
                print(f" ← {result}")
                messages.append({"role":"tool","content":str(result)})

# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent = BookStoreAgent()
    queries = [
        "Suggest some python books",
        "Tell me about Atomic Habits",
        "Give Google books detail for clean code."
    ]
    for query in queries:
        answer = agent.run(query)
        print( f"\nAgent: "f"{answer}")
        print("─" * 60)
