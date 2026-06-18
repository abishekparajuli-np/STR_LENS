import os
import json
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from src.data_loader import load_data
from src.faithfulness import summarize_and_verify


async def process_single_report(record, semaphore, api_key, results_list, file_handle, write_lock):
    """
    Processes a single report under semaphore concurrency limits.

    A write_lock now guards the shared output file handle. write() followed
    by flush() from many concurrent coroutines is not guaranteed atomic --
    under high concurrency (this pipeline runs with concurrency=20) it is
    possible for the event loop to interleave operations between two tasks'
    write/flush pairs, corrupting the JSONL output. The lock makes each
    record's write-then-flush a single atomic unit.
    """
    async with semaphore:
        try:
            res = await summarize_and_verify(record, api_key=api_key)
            results_list.append(res)

            async with write_lock:
                file_handle.write(json.dumps(res) + "\n")
                file_handle.flush()
        except Exception as e:
            str_id = record.get("str_id", "unknown")
            print(f"Error processing record {str_id} ({type(e).__name__}): {e}")
            # Record the failure in the output too, instead of silently
            # dropping it -- otherwise a failed record just vanishes from
            # both the in-memory results list and the output file with no
            # trace, making it impossible to tell "0 missing facts" apart
            # from "this record crashed and was never processed".
            async with write_lock:
                file_handle.write(json.dumps({
                    "str_id": str_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }) + "\n")
                file_handle.flush()


async def run_batch_pipeline(df, output_path, sample_limit=None, concurrency=20):
    """
    Runs the async batch processing pipeline on the dataset.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("WARNING: GROQ_API_KEY is not set (or is the placeholder value). "
              "The pipeline will run in rule-based simulation mode for every record.")

    if sample_limit is not None:
        records = df.head(sample_limit).to_dict("records")
    else:
        records = df.to_dict("records")

    print(f"Starting batch pipeline for {len(records)} records (concurrency limit: {concurrency})...")

    results = []
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        tasks = [
            process_single_report(record, semaphore, api_key, results, f, write_lock)
            for record in records
        ]

        await tqdm.gather(*tasks, desc="Summarizing Reports")

    # Surface a summary of how many records actually succeeded vs failed,
    # since the original code gave no visibility into this at all
    n_errors = sum(1 for r in results if "error" in r)
    n_simulated = sum(1 for r in results if "Simulated" in str(r.get("model", "")))
    print(f"Batch pipeline complete. {len(results)} records processed "
          f"({n_errors} errors, {n_simulated} ran in simulation mode). "
          f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    merged_df, _, _, _ = load_data()

    output_file = "data/sample_outputs/sample_10_reports.jsonl"
    asyncio.run(run_batch_pipeline(merged_df, output_file, sample_limit=10))