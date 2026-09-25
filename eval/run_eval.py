"""LangSmith evaluation with LLM-as-a-judge.

Uploads eval/dataset.jsonl as a LangSmith dataset (once), runs the assistant on every example
with the semantic cache disabled, and scores each answer on correctness, groundedness and
citation presence.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --dataset hr-policy-eval --experiment-prefix gemini-2.5-flash
"""

import argparse
import json
import re
from pathlib import Path

from langsmith import Client
from langsmith.evaluation import evaluate

from hr_rag.config import get_settings
from hr_rag.factory import build_assistant
from hr_rag.llm import build_router

DATASET_FILE = Path(__file__).with_name("dataset.jsonl")

JUDGE_PROMPT = """You are grading an HR policy assistant. Return JSON only: {{"score": <0-1 float>, "reasoning": "<one sentence>"}}.

Criterion: {criterion}

Question: {question}
Reference answer: {reference}
Retrieved policy excerpts:
{contexts}

Assistant answer:
{answer}"""

CORRECTNESS = (
    "Does the assistant answer agree with the reference answer? Score 1 if it is fully "
    "consistent (including correctly declining or refusing when the reference says so), "
    "0.5 if partially correct or missing key details, 0 if wrong or contradictory."
)
GROUNDEDNESS = (
    "Is every factual claim in the answer supported by the retrieved policy excerpts? "
    "Score 1 if fully supported (or the answer makes no policy claims, e.g. a refusal), "
    "0.5 if mostly supported, 0 if it contains unsupported or invented policy details."
)


def ensure_dataset(client: Client, name: str) -> None:
    if client.has_dataset(dataset_name=name):
        return
    dataset = client.create_dataset(name, description="HR policy assistant evaluation set")
    rows = [json.loads(line) for line in DATASET_FILE.read_text().splitlines() if line.strip()]
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": r["question"]} for r in rows],
        outputs=[{"reference": r["reference"]} for r in rows],
    )
    print(f"Created dataset {name!r} with {len(rows)} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hr-policy-eval")
    parser.add_argument("--experiment-prefix", default="hr-rag")
    parser.add_argument("--max-concurrency", type=int, default=2)
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"cache_enabled": False})
    assistant = build_assistant(settings)
    judge = build_router(settings)

    def target(inputs: dict) -> dict:
        r = assistant.answer(inputs["question"])
        return {
            "answer": r.answer,
            "contexts": r.contexts,
            "citations": [c.number for c in r.citations],
            "blocked": r.blocked,
        }

    def llm_judge(criterion: str, inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        prompt = JUDGE_PROMPT.format(
            criterion=criterion,
            question=inputs["question"],
            reference=reference_outputs.get("reference", ""),
            contexts="\n\n".join(outputs.get("contexts", []))[:12000] or "(none)",
            answer=outputs["answer"],
        )
        response = judge.completion(
            model="hr-primary",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", text, re.S)
        data = json.loads(match.group(0)) if match else {"score": 0, "reasoning": text}
        return {"score": float(data.get("score", 0)), "comment": data.get("reasoning", "")}

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        return {"key": "correctness", **llm_judge(CORRECTNESS, inputs, outputs, reference_outputs)}

    def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        return {"key": "groundedness", **llm_judge(GROUNDEDNESS, inputs, outputs, reference_outputs)}

    def has_citation(outputs: dict) -> dict:
        return {"key": "has_citation", "score": 1.0 if outputs.get("citations") else 0.0}

    client = Client()
    ensure_dataset(client, args.dataset)
    results = evaluate(
        target,
        data=args.dataset,
        evaluators=[correctness, groundedness, has_citation],
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
        client=client,
    )
    print(results)


if __name__ == "__main__":
    main()
