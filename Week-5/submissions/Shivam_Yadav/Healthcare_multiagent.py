"""
Week 5 — Multi-Agent Healthcare Research System

"""

import asyncio
import json
import re
import requests

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


MODEL_NAME = "llama3.2"

llm = ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=4096)

INTENT_PROMPT = """You extract structured search parameters from a user's
healthcare research request for three downstream tools:

- disease_stats: needs a country name (or "global")
- drug_info: needs a drug name
- clinical_trials: needs a medical condition

Read the user's request and respond with ONLY a JSON object (no prose, no
markdown fences) with these keys:
  {"country": <string or null>, "drug_name": <string or null>, "condition": <string or null>}

Only include a value if the user actually asked about it. If the user didn't
mention a country, drug, or condition, use null for that key.

Example:
User: "What's the COVID situation in Japan, and can you tell me about ibuprofen?"
{"country": "Japan", "drug_name": "ibuprofen", "condition": null}
"""


# ════════════════════════════════════════════════════════════════════
# REAL-API TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def get_disease_stats(country: str) -> str:
    """Fetch live COVID-19 statistics for a country (or 'global') via disease.sh."""
    try:
        if country.lower() in ("global", "all", "world", "worldwide"):
            url, label = "https://disease.sh/v3/covid-19/all", "Global"
        else:
            url, label = f"https://disease.sh/v3/covid-19/countries/{country}", None
        resp = requests.get(url, timeout=6)
        if resp.status_code == 404:
            return f"Country '{country}' not found."
        resp.raise_for_status()
        data = resp.json()
        name = label or data.get("country", "Unknown")

        def fmt(n):
            return f"{n:,}" if isinstance(n, int) else "N/A"

        return (
            f"COVID-19 Statistics - {name}\n"
            f"  Cases: {fmt(data.get('cases'))}  Deaths: {fmt(data.get('deaths'))}  "
            f"Active: {fmt(data.get('active'))}  Recovered: {fmt(data.get('recovered'))}"
        )
    except requests.RequestException as e:
        return f"Could not fetch disease statistics: {e}"


@tool
def get_drug_info(drug_name: str) -> str:
    """Look up FDA-approved drug label information (OpenFDA API)."""
    def _fetch(field):
        return requests.get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f'openfda.{field}:"{drug_name}"', "limit": 1},
            timeout=8,
        )

    def _trunc(t, n=300):
        return t[:n].rstrip() + "..." if len(t) > n else t

    def _first(v, fb="N/A"):
        return (v[0] if v else fb) if isinstance(v, list) else (v or fb)

    try:
        resp = _fetch("generic_name")
        if resp.status_code == 404:
            resp = _fetch("brand_name")
        if resp.status_code == 404:
            return f"No FDA drug label found for '{drug_name}'."
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return f"No drug label found for '{drug_name}'."
        label, openfda = results[0], results[0].get("openfda", {})
        return (
            f"Drug Info - {', '.join(openfda.get('generic_name', ['N/A']))}\n"
            f"  Brand: {', '.join(openfda.get('brand_name', ['N/A']))}\n"
            f"  Indications: {_trunc(_first(label.get('indications_and_usage')))}\n"
            f"  Warnings: {_trunc(_first(label.get('warnings', label.get('boxed_warning'))))}"
        )
    except requests.RequestException as e:
        return f"Drug info lookup failed: {e}"


@tool
def find_clinical_trials(condition: str, max_results: int = 3) -> str:
    """Search ClinicalTrials.gov for actively recruiting studies on a condition."""
    max_results = max(1, min(max_results, 5))
    try:
        resp = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={
                "query.cond": condition,
                "filter.overallStatus": "RECRUITING",
                "pageSize": max_results,
                "format": "json",
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        studies = data.get("studies", [])
        if not studies:
            return f"No recruiting trials found for '{condition}'."
        lines = [f"Clinical trials for '{condition}' ({data.get('totalCount', 0):,} total):"]
        for i, s in enumerate(studies, 1):
            proto = s.get("protocolSection", {})
            nct = proto.get("identificationModule", {}).get("nctId", "N/A")
            title = proto.get("identificationModule", {}).get("briefTitle", "")[:80]
            lines.append(f"  [{i}] {title} (NCT: {nct})")
        return "\n".join(lines)
    except requests.RequestException as e:
        return f"Clinical trials search failed: {e}"


# ════════════════════════════════════════════════════════════════════
# WORKER 1 — DISEASE STATS AGENT  (its own graph, its own state schema)
# ════════════════════════════════════════════════════════════════════

class DiseaseState(TypedDict):
    country: str
    result: str


def disease_lookup_node(state: DiseaseState) -> dict:
    return {"result": get_disease_stats.invoke({"country": state["country"]})}


def build_disease_graph():
    b = StateGraph(DiseaseState)
    b.add_node("lookup", disease_lookup_node)
    b.add_edge(START, "lookup")
    b.add_edge("lookup", END)
    return b.compile()


# ════════════════════════════════════════════════════════════════════
# WORKER 2 — DRUG INFO AGENT  (its own graph, its own state schema)
# ════════════════════════════════════════════════════════════════════

class DrugState(TypedDict):
    drug_name: str
    result: str


def drug_lookup_node(state: DrugState) -> dict:
    return {"result": get_drug_info.invoke({"drug_name": state["drug_name"]})}


def build_drug_graph():
    b = StateGraph(DrugState)
    b.add_node("lookup", drug_lookup_node)
    b.add_edge(START, "lookup")
    b.add_edge("lookup", END)
    return b.compile()


# ════════════════════════════════════════════════════════════════════
# WORKER 3 — CLINICAL TRIALS AGENT  (its own graph, its own state schema)
# ════════════════════════════════════════════════════════════════════

class TrialsState(TypedDict):
    condition: str
    max_results: int
    result: str


def trials_lookup_node(state: TrialsState) -> dict:
    return {
        "result": find_clinical_trials.invoke(
            {"condition": state["condition"], "max_results": state.get("max_results", 3)}
        )
    }


def build_trials_graph():
    b = StateGraph(TrialsState)
    b.add_node("lookup", trials_lookup_node)
    b.add_edge(START, "lookup")
    b.add_edge("lookup", END)
    return b.compile()


# ════════════════════════════════════════════════════════════════════
# WORKER WRAPPERS 
# ════════════════════════════════════════════════════════════════════

async def run_disease_worker(graph, country: str) -> str:
    out = await graph.ainvoke({"country": country, "result": ""})
    return out["result"]


async def run_drug_worker(graph, drug_name: str) -> str:
    out = await graph.ainvoke({"drug_name": drug_name, "result": ""})
    return out["result"]


async def run_trials_worker(graph, condition: str, max_results: int = 3) -> str:
    out = await graph.ainvoke(
        {"condition": condition, "max_results": max_results, "result": ""}
    )
    return out["result"]


# ════════════════════════════════════════════════════════════════════
# INTENT PARSER (Ollama)
# ════════════════════════════════════════════════════════════════════

async def parse_request(sentence: str) -> dict:
    """Ask the local Ollama model which workers are relevant and with what args."""
    print(f"\n[intent_parser] asking {MODEL_NAME} to parse: \"{sentence}\"")
    messages = [
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=sentence),
    ]
    response = await llm.ainvoke(messages)
    text = response.content if isinstance(response.content, str) else str(response.content)

   
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[intent_parser] could not parse JSON from model output: {text!r}")
        parsed = {"country": None, "drug_name": None, "condition": None}

    print(f"[intent_parser] parsed -> {parsed}")
    return {
        "country": parsed.get("country") or None,
        "drug_name": parsed.get("drug_name") or None,
        "condition": parsed.get("condition") or None,
    }


# ════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════

async def orchestrate(
    disease_graph,
    drug_graph,
    trials_graph,
    request: str,
    simulate_drug_failure: bool = False,
) -> str:
    """
    Orchestrator: uses Ollama to parse a natural-language request into
    structured worker arguments, runs the relevant workers concurrently,
    and combines whatever comes back — successes and failures alike — into
    one report.
    """
    parsed = await parse_request(request)
    country = parsed["country"]
    drug_name = parsed["drug_name"]
    condition = parsed["condition"]

    tasks = []
    task_labels = []

    if country:
        tasks.append(run_disease_worker(disease_graph, country))
        task_labels.append("disease_stats")

    if drug_name:
        if simulate_drug_failure:
            tasks.append(_failing_drug_worker(drug_name))
        else:
            tasks.append(run_drug_worker(drug_graph, drug_name))
        task_labels.append("drug_info")

    if condition:
        tasks.append(run_trials_worker(trials_graph, condition))
        task_labels.append("clinical_trials")

    print(f"\n[orchestrator] fanning out to {len(tasks)} worker(s) in parallel: {task_labels}")

    
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    report_lines = ["MULTI-AGENT RESEARCH REPORT", "=" * 40]
    for label, result in zip(task_labels, raw_results):
        if isinstance(result, Exception):
            print(f"[orchestrator]  ⚠ worker '{label}' failed: {result}")
            report_lines.append(f"\n[{label}] FAILED — {result}\n  (Other workers' results below are still included.)")
        else:
            print(f"[orchestrator]  ✓ worker '{label}' succeeded")
            report_lines.append(f"\n[{label}] SUCCESS\n{result}")

    return "\n".join(report_lines)


async def _failing_drug_worker(drug_name: str) -> str:
    """Wraps the real drug worker but forces a failure, for the failure demo."""
    await asyncio.sleep(0.05)
    raise RuntimeError(f"Simulated FDA API outage while looking up '{drug_name}'")


# ════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════

async def main():
    disease_graph = build_disease_graph()
    drug_graph = build_drug_graph()
    trials_graph = build_trials_graph()

    print("★" * 60)
    print("  Orchestrator: Happy Path (all workers succeed)")
    print("★" * 60)
    report1 = await orchestrate(
        disease_graph, drug_graph, trials_graph,
        request=(
            "Can you check the current COVID-19 numbers for India, tell me "
            "about the medication metformin, and find any ongoing clinical "
            "trials for diabetes?"
        ),
    )
    print(f"\n{'─'*60}\n{report1}")

    print("\n" + "★" * 60)
    print("  Orchestrator: Failure Path (drug worker fails)")
    print("★" * 60)
    report2 = await orchestrate(
        disease_graph, drug_graph, trials_graph,
        request=(
            "I'd like to know how the United States is doing with COVID, "
            "whether metformin has any warnings I should know about, and if "
            "there are any recruiting Alzheimer's trials right now."
        ),
        simulate_drug_failure=True,
    )
    print(f"\n{'─'*60}\n{report2}")


if __name__ == "__main__":
    asyncio.run(main())
