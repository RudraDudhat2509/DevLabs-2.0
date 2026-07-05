# Week 5 — Multi-Agent Healthcare Research System

Step-up from Week 4: instead of one stateful agent, this builds **three
independent worker agents** (each its own compiled LangGraph graph, each
with its own `TypedDict` state schema), coordinated by an **orchestrator**
that fans work out to them in parallel and fans the results back in.

The orchestrator uses **Ollama**  for
one job: reading a natural-language request and figuring out which workers
are relevant and what arguments to give them. The three workers themselves
don't call the LLM — they're plain, fast API calls — so the LLM is used
exactly where it adds value (understanding the request) rather than where
it would just add latency (the lookups themselves).

## How to run

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally
- Internet access (the workers call live public APIs: disease.sh, OpenFDA, ClinicalTrials.gov)

### 2. Pull and start the model
```bash
ollama pull llama3.2
ollama serve       
```
Leave this running in its own terminal (or let the Ollama desktop app handle it).

### 3. Set up a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate      
```

### 4. Install dependencies
```bash
pip install langgraph langchain-core langchain-ollama requests typing_extensions
```

### 5. Run the script
```bash
python week5_multiagent.py
```

This executes, in order:
1. **Happy path** — the orchestrator sends a full sentence to Ollama, which extracts the country/drug/condition, then fans out to all 3 workers in parallel and all succeed.
2. **Failure path** — same flow, but the drug worker is forced to fail; the run still completes with a combined report.

Console output streams `[intent_parser] ...` (the Ollama call and what it
extracted) followed by `[orchestrator] ...` fan-out/fan-in progress and the
final combined report for each run.

### Troubleshooting
- `ConnectionError` / model not responding → confirm `ollama serve` is running and `ollama list` shows `llama3.2`.
- Worker API calls timing out → the public APIs (disease.sh, OpenFDA, ClinicalTrials.gov) are free/unauthenticated and occasionally rate-limit or go down; this is exactly what Sample Run 2 simulates, and a real outage will surface the same way (a `FAILED` entry in the report, not a crash).
- If the model returns prose instead of JSON, `parse_request()` regex-extracts the first `{...}` block and falls back to all-`null` (no workers called) rather than crashing — worth checking the printed `[intent_parser] parsed ->` line if a worker you expected didn't run.
- To change the model, edit `MODEL_NAME` at the top of the file to any model you've pulled (e.g. `llama3.1`, `mistral`).

## Agent topology

```
                         ┌─────────────────────────┐
            sentence ──► │      ORCHESTRATOR        │
                         │  1. Ollama intent parser  │
                         │     (sentence -> args)    │
                         │  2. fixed fan-out/fan-in   │
                         └──────────┬───────────────┘
                                    │ asyncio.gather(..., return_exceptions=True)
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
        ┌──────────────────┐ ┌──────────────┐ ┌─────────────────────┐
        │ Worker 1: DISEASE│ │ Worker 2:    │ │ Worker 3:           │
        │ STATS graph      │ │ DRUG INFO    │ │ CLINICAL TRIALS     │
        │ state: {country, │ │ graph        │ │ graph               │
        │  result}         │ │ state:       │ │ state: {condition,  │
        │ -> disease.sh API│ │ {drug_name,  │ │  max_results,result}│
        │                  │ │  result}     │ │ -> clinicaltrials.gov│
        │                  │ │ -> OpenFDA   │ │                     │
        └──────────────────┘ └──────────────┘ └─────────────────────┘
```

Each worker is its **own `StateGraph`** (`build_disease_graph`,
`build_drug_graph`, `build_trials_graph`) with a state schema scoped only
to what that worker needs — the orchestrator never hands a worker the full
sentence or the other workers' state, only the single field it extracted
for that worker (e.g. just `country`), and only pulls back a single
`result` string from each (state isolation, requirement 4).

## How the Ollama step works

`parse_request()` sends the user's sentence to `llama3.2` via `ChatOllama`
with a system prompt asking for a JSON object of `{country, drug_name,
condition}`, each `null` unless the user actually asked about it. The
response is regex-extracted and `json.loads`-parsed defensively (if the
model adds prose or markdown fences around the JSON, or returns something
unparseable, the orchestrator falls back to calling no workers rather than
crashing). The orchestrator then only builds tasks for the fields that came
back non-null.

## What runs in parallel

In `orchestrate()`, after the intent parse, the disease/drug/trials worker
calls are all built as coroutines first, then run together with a single
`asyncio.gather(*tasks, return_exceptions=True)` call — true fan-out. The
fan-in step iterates over `(label, result)` pairs and builds one combined
report, treating `Exception` instances as failed-but-non-fatal entries.

## Failure handling

`return_exceptions=True` means a worker raising (simulated FDA outage,
real network timeout, malformed API response, etc.) becomes an `Exception`
object in the results list instead of propagating and killing the other
in-flight tasks. The orchestrator inspects each result with
`isinstance(result, Exception)` and reports it as `FAILED` while still
including every worker that did succeed.

## Sample Run 1 — Happy path (all workers succeed)

Request (full sentence, not keywords):
> "Can you check the current COVID-19 numbers for India, tell me about the
> medication metformin, and find any ongoing clinical trials for diabetes?"

```
[intent_parser] asking llama3.2 to parse: "Can you check the current COVID-19 numbers for India, tell me about the medication metformin, and find any ongoing clinical trials for diabetes?"
[intent_parser] parsed -> {'country': 'India', 'drug_name': 'metformin', 'condition': 'diabetes'}

[orchestrator] fanning out to 3 worker(s) in parallel: ['disease_stats', 'drug_info', 'clinical_trials']
[orchestrator]  ✓ worker 'disease_stats' succeeded
[orchestrator]  ✓ worker 'drug_info' succeeded
[orchestrator]  ✓ worker 'clinical_trials' succeeded

────────────────────────────────────────────────────────────
MULTI-AGENT RESEARCH REPORT
========================================

[disease_stats] SUCCESS
COVID-19 Statistics - India
  Cases: 45,035,393  Deaths: 533,570  Active: 44,501,823  Recovered: 0

[drug_info] SUCCESS
Drug Info - SITAGLIPTIN AND METFORMIN HYDROCHLORIDE
  Brand: ZITUVIMET
  Indications: 1 INDICATIONS AND USAGE ZITUVIMET is a combination of sitagliptin, a dipeptidyl peptidase-4 (DPP-4) inhibitor, and metformin hydrochloride (HCl), a biguanide, indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus. ( 1 ) Limitations of Use: Z...
  Warnings: WARNING: LACTIC ACIDOSIS WARNING: LACTIC ACIDOSIS See full prescribing information for complete boxed warning . Postmarketing cases of metformin-associated lactic acidosis have resulted in death, hypothermia, hypotension, and resistant bradyarrhythmias. Symptoms included malaise, myalgias, respirato...

[clinical_trials] SUCCESS
Clinical trials for 'diabetes' (0 total):
  [1] Gestational Diabetes and Perinatal Depression: an Intervention Program (NCT: NCT05800509)
  [2] Achieving Chronic Care equiTy by leVeraging the Telehealth Ecosystem (NCT: NCT06598436)
  [3] Implementation pRogram to Improve Screening and Management for CKD in Diabetes  (NCT: NCT06906640)
```


## Sample Run 2 — Failure & recovery (drug worker fails)

Request (full sentence):
> "I'd like to know how the United States is doing with COVID, whether
> metformin has any warnings I should know about, and if there are any
> recruiting Alzheimer's trials right now."

`simulate_drug_failure=True` forces the drug worker to raise, standing in
for a real-world FDA API outage/timeout.

```
[intent_parser] asking llama3.2 to parse: "I'd like to know how the United States is doing with COVID, whether metformin has any warnings I should know about, and if there are any recruiting Alzheimer's trials right now."
[intent_parser] parsed -> {'country': 'United States', 'drug_name': 'metformin', 'condition': 'alzheimer'}

[orchestrator] fanning out to 3 worker(s) in parallel: ['disease_stats', 'drug_info', 'clinical_trials']
[orchestrator]  ✓ worker 'disease_stats' succeeded
[orchestrator]  ⚠ worker 'drug_info' failed: Simulated FDA API outage while looking up 'metformin'
[orchestrator]  ✓ worker 'clinical_trials' succeeded

────────────────────────────────────────────────────────────
MULTI-AGENT RESEARCH REPORT
========================================

[disease_stats] SUCCESS
COVID-19 Statistics - USA
  Cases: 111,820,082  Deaths: 1,219,487  Active: 786,167  Recovered: 109,814,428

[drug_info] FAILED — Simulated FDA API outage while looking up 'metformin'
  (Other workers' results below are still included.)

[clinical_trials] SUCCESS
Clinical trials for 'Alzheimer's disease' (0 total):
  [1] REXULTI Drug General Use-results Survey (Excessive Motor Activity or Physically)/ (NCT: NCT06875986)
  [2] DC Longitudinal Study on Aging and Specimen Bank (NCT: NCT03702907)
  [3] Use of a Mobile Brain-Body Imaging Approach to Evaluate the Effects of Rhythmic  (NCT: NCT07659964)
```

The run completes and returns a usable report despite one worker failing —
`asyncio.gather(..., return_exceptions=True)` is what makes this possible;
without it, the single `RuntimeError` from the drug worker would have
propagated and killed the disease/trials results too, even though they had
already (or were about to) succeed.

## Mapping to requirements

| Requirement | Where |
|---|---|
| Use Ollama (as in Week 4) | `llm = ChatOllama(model=MODEL_NAME, ...)` + `parse_request()` |
| ≥2 worker agents, separate graphs + state schemas | `build_disease_graph`/`DiseaseState`, `build_drug_graph`/`DrugState`, `build_trials_graph`/`TrialsState` |
| Orchestrator coordinates the workers | `orchestrate()` |
| Parallel fan-out + fan-in | `asyncio.gather(*tasks, return_exceptions=True)` inside `orchestrate()` |
| Isolated state, only final output lifted | `run_disease_worker`/`run_drug_worker`/`run_trials_worker` wrappers pass in only the needed field and return only `result` |
| Failure guard | `return_exceptions=True` + `isinstance(result, Exception)` check in fan-in loop |
