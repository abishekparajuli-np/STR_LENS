import os
import json
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rouge_score import rouge_scorer
from src.data_loader import load_data
from src.extraction import build_fact_checklist, structured_facts_from_record
from src.summarizer import summarize_report
from src.faithfulness import summarize_and_verify, verify_fact_faithfulness

plt.switch_backend("Agg")

_nli_model_cache = {"model": None, "load_failed": False}


def compute_nli_hallucination(summary, narrative):
    """
    Computes an NLI-based hallucination rate (fraction of summary sentences
    that contradict the source narrative) using a cross-encoder NLI model if
    available. Falls back to a word-overlap heuristic otherwise.

    The model is cached at module level instead of being re-loaded on every
    single call -- the original version attempted to load a ~500MB
    cross-encoder from scratch for every single summary being evaluated,
    which is extremely slow and likely to time out or exhaust memory across
    a batch, silently causing every call to fall into the except branch
    regardless of whether sentence-transformers was actually installed.
    """
    if not _nli_model_cache["load_failed"]:
        try:
            if _nli_model_cache["model"] is None:
                from sentence_transformers import CrossEncoder
                _nli_model_cache["model"] = CrossEncoder("cross-encoder/nli-deberta-v3-small")

            model = _nli_model_cache["model"]
            sentences = [s.strip() for s in summary.split(".") if s.strip()]
            if not sentences:
                return 0.0

            pairs = [(narrative, sentence) for sentence in sentences]
            scores = model.predict(pairs)
            contradictions = sum(1 for score in scores if np.argmax(score) == 2)
            return contradictions / len(sentences)
        except Exception as e:
            print(f"NLI model unavailable ({type(e).__name__}: {e}). Using word-overlap heuristic for all subsequent calls.")
            _nli_model_cache["load_failed"] = True

    # Heuristic fallback: words in the summary that don't appear anywhere in
    # the source narrative. This is a weak proxy for hallucination (it will
    # flag paraphrases and synonyms as "hallucinated") and should be
    # reported as such in any write-up, not presented as a true NLI score.
    s_words = set(summary.lower().split())
    n_words = set(narrative.lower().split())
    unmatched = s_words - n_words
    stopwords = {"and", "the", "of", "to", "in", "is", "a", "on", "for", "with", "by", "at", "was", "from"}
    unmatched_substantive = unmatched - stopwords
    if len(s_words) > 0:
        return min(len(unmatched_substantive) / len(s_words), 0.5)
    return 0.0


def build_baseline_a(record):
    """
    Naive heuristic baseline: lists critical fields directly from the
    record with zero LLM cost. Field names match the canonical structured
    facts now produced by data_loader.extract_structured_facts
    (sender_account_number, sender_account_name, sender_institution,
    receiver_account_number, receiver_account_name, receiver_institution),
    not the old guessed names that never matched any real column.
    """
    def first_present(rec, candidates, default="Unknown"):
        for c in candidates:
            val = rec.get(c)
            if val is not None and str(val).strip() and str(val) != "nan":
                return val
        return default

    sender_acc = first_present(record, ["sender_account_number"])
    receiver_acc = first_present(record, ["receiver_account_number"])
    sender_name = first_present(record, ["sender_account_name"], "")
    receiver_name = first_present(record, ["receiver_account_name"], "")
    amount = first_present(record, ["xml_amount_local"], 0.0)
    currency = first_present(record, ["currency_code_local"], "NPR")
    sender_inst = first_present(record, ["sender_institution"])
    receiver_inst = first_present(record, ["receiver_institution"])

    sender_label = f"{sender_name} ({sender_acc})" if sender_name else sender_acc
    receiver_label = f"{receiver_name} ({receiver_acc})" if receiver_name else receiver_acc

    return (
        f"AML transaction alert. Parties: Sender {sender_label} at {sender_inst}, "
        f"Receiver {receiver_label} at {receiver_inst}. Amount {currency} {amount}."
    )


async def run_ablation_study(df, sample_limit=5):
    """
    Runs Baseline A, Baseline B, and STR-Lens on a sample of reports and computes metrics.
    """
    records = df.head(sample_limit).to_dict("records")
    api_key = os.environ.get("GROQ_API_KEY")

    results = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for idx, r in enumerate(records):
        print(f"Evaluating record {idx+1}/{sample_limit} ({r.get('str_id')})...")
        narrative = r.get("narrative_text", "")
        structured = structured_facts_from_record(r)
        facts = build_fact_checklist(narrative, structured)

        # 1. Gold reference summary: full pipeline with fact checklist
        gold_res = await summarize_report(r, facts, api_key=api_key)
        gold_summary = gold_res["summary_text"]

        # 2. Baseline A: naive heuristic, zero LLM
        baseline_a = build_baseline_a(r)

        # 3. Baseline B: basic LLM prompt, no checklist injection, no verification
        baseline_b_res = await summarize_report(r, {}, api_key=api_key)
        baseline_b = baseline_b_res["summary_text"]

        # 4. STR-Lens: full pipeline with verification/re-prompt loop
        pipeline_res = await summarize_and_verify(r, api_key=api_key)
        str_lens = pipeline_res["summary_text"]

        for system_name, summary in [("Baseline A (Naive Heuristic)", baseline_a),
                                      ("Baseline B (Basic LLM)", baseline_b),
                                      ("STR-Lens (Full Pipeline)", str_lens)]:

            r_score = scorer.score(gold_summary, summary)["rougeL"].fmeasure
            faith_score, missing = verify_fact_faithfulness(summary, facts)
            halluc_rate = compute_nli_hallucination(summary, narrative)
            compression = 1.0 - (len(summary.split()) / max(len(narrative.split()), 1))

            results.append({
                "record_idx": idx,
                "str_id": r.get("str_id"),
                "system": system_name,
                "rouge_l": r_score,
                "faithfulness": faith_score,
                "hallucination_rate": halluc_rate,
                "compression_ratio": compression,
                "word_count": len(summary.split())
            })

    return pd.DataFrame(results)


def generate_dashboard(eval_df, output_path="results/metrics_dashboard.png"):
    """
    Generates comparative bar plots and saves to path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    summary_df = eval_df.groupby("system")[["rouge_l", "faithfulness", "hallucination_rate", "compression_ratio"]].mean().reset_index()
    print("\nEvaluation Summary Metrics:")
    print(summary_df.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.barplot(data=summary_df, x="system", y="rouge_l", ax=axes[0, 0], hue="system", palette="Blues_d", legend=False)
    axes[0, 0].set_title("ROUGE-L Score (higher is better)")
    axes[0, 0].set_ylim(0, 1.0)

    sns.barplot(data=summary_df, x="system", y="faithfulness", ax=axes[0, 1], hue="system", palette="Greens_d", legend=False)
    axes[0, 1].set_title("Faithfulness Checklist Score (higher is better)")
    axes[0, 1].set_ylim(0, 1.0)

    sns.barplot(data=summary_df, x="system", y="hallucination_rate", ax=axes[1, 0], hue="system", palette="Oranges_d", legend=False)
    axes[1, 0].set_title("NLI Hallucination Rate (lower is better)")
    axes[1, 0].set_ylim(0, 0.5)

    sns.barplot(data=summary_df, x="system", y="compression_ratio", ax=axes[1, 1], hue="system", palette="Purples_d", legend=False)
    axes[1, 1].set_title("Text Compression Ratio (higher is better)")
    axes[1, 1].set_ylim(0, 1.0)

    for ax in axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved dashboard chart to {output_path}")


if __name__ == "__main__":
    merged_df = pd.read_parquet("data/clean_data.parquet")

    eval_results = asyncio.run(run_ablation_study(merged_df, sample_limit=5))

    Path("results").mkdir(parents=True, exist_ok=True)
    eval_results.to_csv("results/eval_report.csv", index=False)
    print("Saved evaluation records to results/eval_report.csv")

    generate_dashboard(eval_results)