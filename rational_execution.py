import asyncio
import concurrent.futures
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from datasets import load_dataset, Dataset
from tqdm import tqdm

from camel.verifiers.python_verifier import PythonVerifier
from camel.verifiers.models import VerificationOutcome

# Configuration constants
# -----------------------

# Dataset configuration
DEFAULT_DATASET_NAME = "zjrwtxtechstudio/testbiology01"
DEFAULT_CACHE_DIR = "cached_datasets"
DEFAULT_OUTPUT_DIR = "loong_results"

# Execution configuration
DEFAULT_MAX_WORKERS = 40
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 60.0

# Required packages for the Python verifier
REQUIRED_PACKAGES = [
    "numpy", 
    "pandas", 
    "matplotlib", 
    "biopython", 
    "scipy", 
    "sympy", 
    "networkx", 
    "Crypto", 
    "cryptography", 
    "gmpy2", 
    "QuantLib", 
    "pyscipopt", 
    "gurobipy", 
    "seaborn", 
    "cvxpy", 
    "plotly", 
    "scikit-learn"
]

# Global variable to store the verifier in each process
_PROCESS_VERIFIER = None

async def get_or_create_verifier() -> PythonVerifier:
    """
    Get an existing verifier or create a new one if it doesn't exist.
    This ensures one verifier per process.
    
    Returns:
        A PythonVerifier instance.
    """
    global _PROCESS_VERIFIER
    
    if _PROCESS_VERIFIER is None:
        # Initialize the Python verifier with required packages
        _PROCESS_VERIFIER = PythonVerifier(timeout=DEFAULT_TIMEOUT, required_packages=REQUIRED_PACKAGES)
        
        # Setup the virtual environment once per process
        await _PROCESS_VERIFIER._setup(uv=True)
        
    return _PROCESS_VERIFIER


async def cleanup_verifier():
    """
    Clean up the verifier if it exists.
    """
    global _PROCESS_VERIFIER
    
    if _PROCESS_VERIFIER is not None and _PROCESS_VERIFIER.venv_path:
        await _PROCESS_VERIFIER._cleanup()
        _PROCESS_VERIFIER = None


async def execute_rationale(
    rationale: str, 
    verifier: PythonVerifier
) -> Dict[str, Any]:
    """
    Execute a rationale using the Python verifier.

    Args:
        rationale: The Python code to execute.
        verifier: The PythonVerifier instance to use for execution.

    Returns:
        Dictionary containing execution results.
    """
    # Preprocess the rationale to fix common syntax issues
    preprocessed_rationale = rationale
    
    try:
        # Execute the rationale
        result = await verifier._verify_implementation(preprocessed_rationale, None)
        
        return {
            "status": result.status.name,
            "result": result.result,
            "error_message": result.error_message,
            "execution_successful": result.status == VerificationOutcome.SUCCESS,
            "preprocessed_code": preprocessed_rationale
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "result": "",
            "error_message": str(e),
            "execution_successful": False,
            "preprocessed_code": preprocessed_rationale
        }


async def process_rationale(rationale: str) -> Dict[str, Any]:
    """
    Process a rationale using the process-wide verifier.
    
    Args:
        rationale: The Python code to execute.
        
    Returns:
        Dictionary containing execution results.
    """
    # Get or create the process-wide verifier
    verifier = await get_or_create_verifier()
    
    # Execute the rationale
    return await execute_rationale(rationale, verifier)


def process_single_sample(
    idx_item_tuple: Tuple[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process a single sample from the dataset.
    
    Args:
        idx_item_tuple: Tuple containing the index and item from the dataset.
        
    Returns:
        Dictionary containing the processed result.
    """
    idx, item = idx_item_tuple
    
    # Extract the rationale
    rationale = item.get("rationale", "")
    if not rationale:
        print(f"Warning: Sample {idx} has no rationale. Skipping.")
        return None
    
    # Execute the rationale using asyncio in this process
    execution_result = asyncio.run(process_rationale(rationale))
    
    # Get source_type from dataset or use domain as fallback
    source_type = item.get("source_type", item.get("domain", ""))
    
    # Create a result entry
    return {
        "id": idx,
        "question": item.get("question", ""),
        "domain": item.get("domain", ""),
        "source_type": source_type,
        "rationale": rationale,
        "execution_result": execution_result
    }


def process_batch(items_batch: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Process a batch of samples using a single verifier.
    
    Args:
        items_batch: List of tuples containing the index and item from the dataset.
        
    Returns:
        List of processed results.
    """
    results = []
    
    try:
        for idx_item in items_batch:
            result = process_single_sample(idx_item)
            if result:
                results.append(result)
    finally:
        # Clean up the verifier at the end of batch processing
        asyncio.run(cleanup_verifier())
        
    return results


async def process_split(
    split_name: str,
    data: Union[Dataset, Any],
    output_dir: str,
    max_samples: Optional[int] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Dict[str, Any]:
    """
    Process a single split from the dataset.
    
    Args:
        split_name: Name of the split to process.
        data: Dataset split to process.
        output_dir: Directory to save output files.
        max_samples: Maximum number of samples to process (None for all).
        max_workers: Maximum number of worker processes to use.
        batch_size: Number of samples to process in each worker batch.
        
    Returns:
        Dictionary containing summary statistics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output file path
    output_file = os.path.join(output_dir, f"loong_execution_results_{split_name}.json")
    
    # Limit the number of samples if specified
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
    
    # Prepare data for parallel processing
    items_to_process = [(idx, item) for idx, item in enumerate(data)]
    
    # Create batches of items
    batches = []
    for i in range(0, len(items_to_process), batch_size):
        batches.append(items_to_process[i:i + batch_size])
    
    results = []
    
    print(f"Processing {len(data)} samples from split '{split_name}' with {max_workers} workers and batch size {batch_size}...")
    start_time = time.time()
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batches for processing
        future_to_batch = {
            executor.submit(process_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(batches)):
            batch_results = future.result()
            results.extend(batch_results)
    
    end_time = time.time()
    
    # Sort results by ID to maintain original order
    results.sort(key=lambda x: x["id"])
    
    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results for split '{split_name}' saved to {output_file}")
    
    # Generate summary statistics
    successful = sum(1 for r in results if r["execution_result"]["execution_successful"])
    
    summary = {
        "split_name": split_name,
        "total_samples": len(results),
        "successful_executions": successful,
        "success_rate": successful/len(results)*100 if results else 0,
        "failed_executions": len(results) - successful,
        "failure_rate": (len(results) - successful)/len(results)*100 if results else 0,
        "total_processing_time": end_time - start_time,
        "average_time_per_sample": (end_time - start_time)/len(results) if results else 0
    }
    
    # Print summary
    print(f"Execution summary for split '{split_name}':")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Successful executions: {summary['successful_executions']} ({summary['success_rate']:.2f}%)")
    print(f"  Failed executions: {summary['failed_executions']} ({summary['failure_rate']:.2f}%)")
    print(f"  Total processing time: {summary['total_processing_time']:.2f} seconds")
    print(f"  Average time per sample: {summary['average_time_per_sample']:.2f} seconds")
    
    return summary


async def process_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_samples: Optional[int] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    splits: Optional[List[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> None:
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Prepare local dataset directory path
    local_dataset_dir = os.path.join(cache_dir, dataset_name.replace("/", "_"))
    
    # Download and save the dataset if needed
    local_dataset = download_and_prepare_dataset(dataset_name, cache_dir, local_dataset_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all available splits
    available_splits = list(local_dataset.keys())
    print(f"Available splits in local dataset: {available_splits}")
    
    # Determine which splits to process
    splits_to_process = splits if splits else available_splits
    
    # Process each split
    all_summaries = []
    for split_name in splits_to_process:
        if split_name not in available_splits:
            print(f"Warning: Split '{split_name}' not found in local dataset. Skipping.")
            continue
        
        # Process the split
        summary = await process_split(
            split_name=split_name,
            data=local_dataset[split_name],
            output_dir=output_dir,
            max_samples=max_samples,
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        all_summaries.append(summary)
    
    # Save overall summary to JSON file
    summary_file = os.path.join(output_dir, "execution_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    
    print(f"Overall execution summary saved to {summary_file}")
    
    # Print overall summary
    total_samples = sum(s["total_samples"] for s in all_summaries)
    total_successful = sum(s["successful_executions"] for s in all_summaries)
    total_time = sum(s["total_processing_time"] for s in all_summaries)
    
    print("\nOverall Execution Summary:")
    print(f"  Total splits processed: {len(all_summaries)}")
    print(f"  Total samples processed: {total_samples}")
    print(f"  Total successful executions: {total_successful} ({total_successful/total_samples*100:.2f}%)")
    print(f"  Total failed executions: {total_samples - total_successful} ({(total_samples - total_successful)/total_samples*100:.2f}%)")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Average time per sample: {total_time/total_samples:.2f} seconds")


# Helper functions for dataset handling
# ---------------------------------

def download_and_prepare_dataset(dataset_name: str, cache_dir: str, local_dataset_dir: str) -> Dict[str, Dataset]:
    """Download a dataset from Hugging Face and prepare it for local use.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        cache_dir: Directory to cache the downloaded dataset
        local_dataset_dir: Directory to save the processed dataset
        
    Returns:
        Dictionary of dataset splits
    """
    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(local_dataset_dir, exist_ok=True)
    
    # Download the dataset first
    print(f"Downloading dataset: {dataset_name} to {cache_dir}")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    print(f"Saving dataset locally to: {local_dataset_dir}")
    for split_name in dataset.keys():
        split_path = os.path.join(local_dataset_dir, f"{split_name}.json")
        if not os.path.exists(split_path):
            # Convert to list and save as JSON
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(dataset[split_name].to_dict(), f, indent=2, ensure_ascii=False)
            print(f"Saved {split_name} split to {split_path}")
        else:
            print(f"Found existing {split_name} split at {split_path}")
    
    # Now load the local dataset for processing
    print(f"Using local dataset from: {local_dataset_dir}")
    
    # Load the local dataset
    local_dataset = {}
    for file in os.listdir(local_dataset_dir):
        if file.endswith(".json"):
            split_name = file.replace(".json", "")
            with open(os.path.join(local_dataset_dir, file), 'r', encoding='utf-8') as f:
                split_data = json.load(f)
                # Convert the JSON format back to a dataset-like format
                local_dataset[split_name] = Dataset.from_dict(split_data)
                
    return local_dataset


# Command-line interface
# ---------------------

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Execute rationales from the Loong dataset")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME, 
                        help="Hugging Face dataset name")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for JSON files")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process per split")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Number of worker processes to use for parallel execution")
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma-separated list of specific splits to process (default: all splits)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of samples to process in each worker batch")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Directory to cache the downloaded dataset")
    
    return parser.parse_args()


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Convert comma-separated splits to list
    splits = [split.strip() for split in args.splits.split(",")] if args.splits else None
    
    # Run the main function
    asyncio.run(process_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_workers=args.workers,
        splits=splits,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir
    ))
