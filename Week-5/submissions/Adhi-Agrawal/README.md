# Bookstore Agent System

## Overview
This project implements a **Bookstore Agent System** using **LangGraph** and **Ollama**. A Supervisor coordinates two independent worker agents. The workers execute in parallel using `asyncio.gather()`, and their outputs are merged to generate the final response.
---

## Agent Topology
* **Supervisor** receives the customer query.
* The Supervisor splits the query into **Catalog Task** and **Recommendation Task**.
* **Catalog Worker** extracts search filters and searches `books.csv`.
* **Recommendation Worker** generates book recommendations, verifies them, and retries automatically if verification fails.
* Both workers run **in parallel** using `asyncio.gather()`.
* The Supervisor combines both worker outputs and returns the final response.
---

## Parallel Execution
The Supervisor runs the **Catalog Worker** and **Recommendation Worker** in parallel using `asyncio.gather()`. After both workers complete their tasks, the Supervisor combines their outputs into a single final response. If one worker fails, the other continues running using `return_exceptions=True`.
---

## Sample Run 1 (Happy Path)

**Customer**
```text
do you have atomic habits?
```

**Execution**
```text
[Supervisor] Understanding customer request...
Catalog Task : Find the book 'Atomic Habits'
Recommendation Task :
[Catalog Worker] Extracting search filters...
[Catalog Worker] Searching books.csv...
[Supervisor] Preparing final response...
```

**Assistant**
```text
We do have "Atomic Habits" by James Clear.
Stock: 12 copies
Price: ₹699
```

---

## Sample Run 2 (Worker Failure Recovery)

**Customer**
```text
recommend any programming book
```

**Execution**
```text
[Supervisor] Understanding customer request...
Catalog Task : Find programming books
Recommendation Task : Recommend programming books
[Catalog Worker] Extracting search filters...
[Recommendation Worker] Generating recommendations...
[Recommendation Worker] Recommended book not found. Retrying...
[Recommendation Worker] Generating recommendations...
[Recommendation Worker] Recommendation verified.
[Catalog Worker] Searching books.csv...
[Supervisor] Preparing final response...
```

**Assistant**
```text
Programming books available:
• Clean Code (In Stock)
Python Crash Course is currently out of stock.
Recommendation Worker retried and successfully returned an available recommendation.
```
---

