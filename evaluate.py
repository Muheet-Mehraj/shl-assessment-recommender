"""
Offline evaluator — tests the agent logic against the 10 reference conversations.
Runs without a live server by calling the LLM directly via the same code path.

Usage:
    ANTHROPIC_API_KEY=sk-... python evaluate.py
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from main import call_gemini as call_llm, parse_llm_response, validate_recommendations

# ── Ground-truth final shortlists from the 10 conversations ──────────
TRACES = [
    {
        "id": "C1 - Senior Leadership Selection",
        "final_query": [
            {"role": "user", "content": "We need a solution for senior leadership."},
            {"role": "assistant", "content": '{"reply":"...","recommendations":null,"end_of_conversation":false}'},
            {"role": "user", "content": "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."},
            {"role": "assistant", "content": '{"reply":"...","recommendations":null,"end_of_conversation":false}'},
            {"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."},
        ],
        "expected_names": [
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ],
    },
    {
        "id": "C2 - Senior Rust Engineer",
        "final_query": [
            {"role": "user", "content": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"},
            {"role": "assistant", "content": '{"reply":"SHL catalog has no Rust-specific test...","recommendations":null,"end_of_conversation":false}'},
            {"role": "user", "content": "Yes, go ahead. Should I also add a cognitive test for this level?"},
        ],
        "expected_names": [
            "Smart Interview Live Coding",
            "Linux Programming (General)",
            "Networking and Implementation (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C3 - Contact Centre Agents",
        "final_query": [
            {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. English, US accent. Two-stage: new simulation for volume, old for finalists."},
        ],
        "expected_names": [
            "SVAR Spoken English (US) (New)",
            "Contact Center Call Simulation (New)",
            "Entry Level Customer Serv - Retail & Contact Center",
            "Customer Service Phone Simulation",
        ],
    },
    {
        "id": "C4 - Graduate Financial Analysts",
        "final_query": [
            {"role": "user", "content": "Hiring graduate financial analysts — final-year students, no work experience. Numerical reasoning, finance knowledge, and situational judgement. Two-stage: numerical + graduate scenarios first, then domain tests for shortlisted."},
        ],
        "expected_names": [
            "SHL Verify Interactive – Numerical Reasoning",
            "Financial Accounting (New)",
            "Basic Statistics (New)",
            "Graduate Scenarios",
        ],
    },
    {
        "id": "C5 - Sales Organisation Talent Audit",
        "final_query": [
            {"role": "user", "content": "Annual talent audit to re-skill our Sales organization. OPQ for everyone, MQ only where we want motivators. Five solutions as audit stack."},
        ],
        "expected_names": [
            "Global Skills Assessment",
            "Global Skills Development Report",
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ MQ Sales Report",
            "Sales Transformation 2.0 - Individual Contributor",
        ],
    },
    {
        "id": "C6 - Chemical Plant Operators",
        "final_query": [
            {"role": "user", "content": "Hiring plant operators for a chemical facility. Safety is top priority. We're industrial — the 8.0 bundle is the right fit."},
        ],
        "expected_names": [
            "Manufac. & Indust. - Safety & Dependability 8.0",
            "Workplace Health and Safety (New)",
        ],
    },
    {
        "id": "C7 - Bilingual Healthcare Admin",
        "final_query": [
            {"role": "user", "content": "Bilingual healthcare admin staff in South Texas. HIPAA critical. Functionally bilingual — English fluent for written work. Hybrid battery."},
        ],
        "expected_names": [
            "HIPAA (Security)",
            "Medical Terminology (New)",
            "Microsoft Word 365 - Essentials (New)",
            "Dependability and Safety Instrument (DSI)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C8 - Admin Assistants Excel/Word",
        "final_query": [
            {"role": "user", "content": "I need to quickly screen admin assistants for Excel and Word daily use. Include simulation to capture capabilities."},
        ],
        "expected_names": [
            "Microsoft Excel 365 (New)",
            "Microsoft Word 365 (New)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C9 - Senior Full-Stack Engineer",
        "final_query": [
            {"role": "user", "content": "Senior IC backend Java engineer. Core Java Advanced, Spring, SQL, AWS, Docker. Keep Verify G+. Lock it in."},
        ],
        "expected_names": [
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    {
        "id": "C10 - Graduate Management Trainee",
        "final_query": [
            {"role": "user", "content": "Graduate management trainee scheme. Full battery: cognitive and situational judgement. Drop the OPQ. Final: Verify G+ and Graduate Scenarios."},
        ],
        "expected_names": [
            "SHL Verify Interactive G+",
            "Graduate Scenarios",
        ],
    },
]


async def evaluate_trace(trace: dict) -> dict:
    messages = trace["final_query"]
    raw = await call_llm(messages)
    parsed = parse_llm_response(raw)
    recs = validate_recommendations(parsed.get("recommendations")) or []
    rec_names = {r.name for r in recs}
    expected = set(trace["expected_names"])

    hits = rec_names & expected
    recall = len(hits) / len(expected) if expected else 1.0
    precision = len(hits) / len(rec_names) if rec_names else 0.0

    return {
        "id": trace["id"],
        "recall@10": round(recall, 3),
        "precision": round(precision, 3),
        "hits": sorted(hits),
        "missed": sorted(expected - rec_names),
        "extra": sorted(rec_names - expected),
        "recommended": sorted(rec_names),
    }


async def main():
    print("=" * 70)
    print("SHL Agent Evaluation — 10 Reference Conversations")
    print("=" * 70)

    results = []
    for trace in TRACES:
        print(f"\n▶ {trace['id']} ...", end=" ", flush=True)
        result = await evaluate_trace(trace)
        results.append(result)
        print(f"Recall@10={result['recall@10']:.3f}  Precision={result['precision']:.3f}")
        if result["missed"]:
            print(f"  MISSED: {result['missed']}")
        if result["extra"]:
            print(f"  EXTRA:  {result['extra']}")

    avg_recall = sum(r["recall@10"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)

    print("\n" + "=" * 70)
    print(f"Average Recall@10 : {avg_recall:.3f}")
    print(f"Average Precision : {avg_precision:.3f}")
    print("=" * 70)

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to eval_results.json")


if __name__ == "__main__":
    asyncio.run(main())
