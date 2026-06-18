import os
import json
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from src.data_loader import load_data
from src.faithfulness import summarize_and_verify

async def process_single_report(record, semaphore, api_key, results_list, file_handle):
    """
    Processes a single report under semaphore concurrency limits.
    """
    async with semaphore:
        try:
            # We run the summarisation and faithfulness loop
            res = await summarize_and_verify(record, api_key=api_key)
            results_list.append(res)
            
            # Write to output file immediately to survive interruption
            file_handle.write(json.dumps(res) + "\n")
            file_handle.flush()
        except Exception as e:
            print(f"Error processing record {record.get('str_id', 'unknown')}: {e}")

async def run_batch_pipeline(df, output_path, sample_limit=None, concurrency=20):
    """
    Runs the async batch processing pipeline on the dataset.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Apply sample limit if specified
    if sample_limit is not None:
        records = df.head(sample_limit).to_dict("records")
    else:
        records = df.to_dict("records")
        
    print(f"Starting batch pipeline for {len(records)} records (concurrency limit: {concurrency})...")
    
    results = []
    semaphore = asyncio.Semaphore(concurrency)
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        tasks = [
            process_single_report(record, semaphore, api_key, results, f)
            for record in records
        ]
        
        # Use tqdm to show a progress bar
        await tqdm.gather(*tasks, desc="Summarizing Reports")
        
    print(f"Batch pipeline complete. Results saved to {output_path}")
    return results

if __name__ == "__main__":
    # Load dataset
    merged_df, _, _, _ = load_data()
    
    # Run on a sample of 10 reports as specified in Phase 3 deliverables
    output_file = "data/sample_outputs/sample_10_reports.jsonl"
    asyncio.run(run_batch_pipeline(merged_df, output_file, sample_limit=10))
