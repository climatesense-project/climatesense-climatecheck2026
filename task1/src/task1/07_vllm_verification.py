"""Claim verification using a local vLLM server (Task 1.2).

Prompts a locally hosted LLM to classify claim-abstract pairs as
Supports / Refutes / Not Enough Information.

Expects a vLLM-compatible OpenAI API server at VLLM_BASE_URL (default: http://localhost:8000).

Usage:
    python 07_vllm_verification.py
    python 07_vllm_verification.py --input artifacts/ensemble_top10_test.csv --top-n 5
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from json_repair import loads, repair_json
from tqdm import tqdm

from task1 import config

LABEL_MAP = {
    "supports": "Supports",
    "refutes": "Refutes",
    "not_enough_information": "Not Enough Information",
}

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 500},
        "classification": {
            "type": "string",
            "enum": ["supports", "refutes", "not_enough_information"],
        },
    },
    "required": ["reasoning", "classification"],
    "additionalProperties": False,
}


def get_verification_prompt(
    claim: str,
    abstract: str,
) -> str:
    """Construct a prompt for claim verification."""
    prompt_parts = [
        """You are a strict Natural Language Inference (NLI) classifier.

Task:
Determine whether the ABSTRACT:
- supports the CLAIM,
- refutes the CLAIM, or
- provides not_enough_information.

Rules:
- Use ONLY information explicitly stated in the abstract.
- Do NOT use outside knowledge.
- Topic similarity alone does NOT imply support.
- Absence of evidence is NOT refutation.
- Only choose refutes if the abstract directly contradicts the claim.
- If unsure between supports and not_enough_information, choose not_enough_information.""".strip()
    ]

    prompt_parts.append(
        f"""
### Now classify:

Claim:
{claim.strip()[:4000]}

Abstract:
{abstract.strip()[:4000]}

Respond with ONLY valid JSON matching this exact schema: {json.dumps(CLASSIFICATION_SCHEMA)}
No markdown, commentary, or extra text.""".strip()
    )

    return "\n\n".join(prompt_parts)


def call_vllm(prompt: str, retries: int = 3) -> dict[str, str]:
    """vLLM with full robustness."""
    fallback = {
        "classification": "not_enough_information",
        "reasoning": "Technical failure: truncated/invalid response after retries.",
    }

    for attempt in range(retries):
        try:
            headers = {"Content-Type": "application/json"}
            temp = 0.0 if attempt == 0 else min(0.1 * (attempt + 1), 0.3)
            max_toks = int(512 * (1.5**attempt))

            body = {
                "model": config.VLLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": max_toks,
                "extra_body": {"guided_json": CLASSIFICATION_SCHEMA},
            }

            resp = requests.post(
                f"{config.VLLM_BASE_URL}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = (
                choice["message"]["content"].strip() if "message" in choice else ""
            )
            finish_reason = choice.get("finish_reason", "unknown")

            print(
                f"[vLLM #{attempt + 1}] temp={temp}, toks={max_toks}, finish={finish_reason}"
            )
            if content:
                if len(content) <= 150:
                    print(f"Preview: {content!r}")
                else:
                    head = content[:75]
                    tail = content[-75:]
                    print(f"Preview: {head!r} ... {tail!r}")
            else:
                print("Preview: ''")

            # Save raw response for debugging
            raw_file = config.ARTIFACTS_DIR / "vllm_verification_raw_responses.jsonl"
            with open(raw_file, "a") as f:
                json.dump(content, f)
                f.write("\n")

            if not content:
                print("Empty.")
                continue

            if finish_reason == "length":
                print("Truncated → retry.")

            fixed = repair_json(content)
            parsed = loads(fixed)

            if (
                isinstance(parsed, dict)
                and parsed.get("classification")
                in ["supports", "refutes", "not_enough_information"]
                and isinstance(parsed.get("reasoning"), str)
            ):
                print("✅ Success!")
                return parsed

            print(f"❌ Invalid: {fixed[:200]}")

        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            print(f"Error #{attempt + 1}: {str(e)[:100]}")

        if attempt < retries - 1:
            wait = min(2**attempt, 10)
            print(f"Retry in {wait}s...")
            time.sleep(wait)

    print("→ Fallback.")
    return fallback


def verify_claim(
    claim: str,
    abstract: str,
) -> dict[str, str]:
    """Verify a single claim-abstract pair."""
    prompt = get_verification_prompt(claim, abstract)
    return call_vllm(prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=config.ENSEMBLE_TOP_N)
    args = parser.parse_args()

    # Load retrieval results
    if args.input is not None:
        retrieval_path = args.input
    else:
        retrieval_path = config.ARTIFACTS_DIR / f"ensemble_top{args.top_n}_test.csv"
    if not retrieval_path.exists():
        print(
            f"Retrieval file not found: {retrieval_path}. Run 06_ensemble_reranking.py first."
        )
        return

    df_retrieval = pd.read_csv(retrieval_path)
    total_rows = len(df_retrieval)
    print(f"Loaded retrieval results: {retrieval_path} ({total_rows} rows)")
    df_retrieval = df_retrieval[df_retrieval["rank"] <= args.top_n]
    if len(df_retrieval) != total_rows:
        print(f"After filtering to top {args.top_n}: {len(df_retrieval)} rows")

    df_test = pd.read_parquet(config.TEST_PATH)
    corpus = pd.read_parquet(config.CORPUS_PATH)
    abstract_lookup = dict(zip(corpus["abstract_id"], corpus["abstract"], strict=True))

    # Build pairs
    rows = []
    for _, r in df_retrieval.iterrows():
        claim_text = df_test.loc[df_test["claim_id"] == r["claim_id"], "claim"].values[
            0
        ]
        abstract_text = abstract_lookup.get(r["abstract_id"], "")
        rows.append(
            {
                "claim_id": r["claim_id"],
                "abstract_id": r["abstract_id"],
                "rank": r["rank"],
                "claim_text": claim_text,
                "abstract_text": abstract_text,
            }
        )
    df_pairs = pd.DataFrame(rows)

    output_path = config.ARTIFACTS_DIR / f"verification_vllm_top{args.top_n}.csv"
    done_pairs = set()
    if output_path.exists():
        df_done = pd.read_csv(output_path)
        done_pairs = set(zip(df_done["claim_id"], df_done["abstract_id"], strict=True))
        print(f"Resuming from {len(done_pairs)} completed pairs.")

    df_pairs = df_pairs[
        ~df_pairs.apply(
            lambda r: (r["claim_id"], r["abstract_id"]) in done_pairs, axis=1
        )
    ]

    print(f"vLLM server: {config.VLLM_BASE_URL}  |  model: {config.VLLM_MODEL}")
    print(f"Pairs to process: {len(df_pairs)}")

    for _, row in tqdm(
        df_pairs.iterrows(),
        desc=f"vLLM verify (top {args.top_n})",
        total=len(df_pairs),
    ):
        claim_text = row["claim_text"]
        abstract_text = row["abstract_text"]

        result = verify_claim(claim_text, abstract_text)

        label = result["classification"]  # "supports" etc.
        reasoning = result["reasoning"]

        print(f"Claim {row['claim_id']} - Abs {row['abstract_id']} → {label}")
        print(f"  Reasoning: {reasoning[:100]}...")

        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["claim_id", "abstract_id", "rank", "label"],
            )
            if f.tell() == 0:
                writer.writeheader()
            row_out = {
                "claim_id": row["claim_id"],
                "abstract_id": row["abstract_id"],
                "rank": row["rank"],
                "label": LABEL_MAP.get(label, label),
            }
            writer.writerow(row_out)
            f.flush()
            with contextlib.suppress(Exception):
                os.fsync(f.fileno())

        done_pairs.add((row["claim_id"], row["abstract_id"]))

    print(f"Saved {len(done_pairs)} verifications: {output_path}")


if __name__ == "__main__":
    main()
