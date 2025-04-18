import asyncio
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from camel.verifiers.python_verifier import PythonVerifier
from camel.verifiers.models import VerificationOutcome
from camel.verifiers import MathVerifier
from physic_verifier_tem import PhysicsVerifier
import logging

# Configuration constants
DEFAULT_MAX_WORKERS = 6
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 60.0
DEFAULT_CONCURRENT_BATCHES = 5  # Number of batches to process concurrently
ENV_CACHE_ENABLED = True        # Enable caching of virtual environments

logger = logging.getLogger(__name__)

async def setup_verifier(required_packages: List[str], timeout: float = 60.0) -> PythonVerifier:
    """
    Set up a Python verifier with the required packages.
    
    Args:
        required_packages: List of required packages with versions.
        timeout: Timeout for code execution in seconds.
        
    Returns:
        A configured PythonVerifier instance.
    """
    # 确保medcalc-bench被安装，无论是否在required_packages列表中
    if "medcalc-bench" not in required_packages:
        required_packages = required_packages + ["medcalc-bench"]
    
    verifier = PythonVerifier(timeout=timeout, required_packages=required_packages)
    await verifier.setup(uv=True)
    return verifier


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
    try:
        # Execute the rationale
        result = await verifier.verify(rationale, None)
        
        return {
            "status": result.status.name,
            "result": result.result,
            "error_message": result.error_message,
            "execution_successful": result.status == VerificationOutcome.SUCCESS
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "result": "",
            "error_message": str(e),
            "execution_successful": False
        }


async def compare_results(execution_result: str, final_answer: str, domain: str = None) -> bool:
    """
    Enhanced comparison between execution result and final answer.
    Performs normalization before comparison for more accurate matching.
    For advanced_math domain, uses MathVerifier for more sophisticated comparison.
    
    Args:
        execution_result: The result from code execution.
        final_answer: The expected final answer.
        domain: The problem domain (e.g., 'advanced_math').
        
    Returns:
        True if the results match, False otherwise.
    """
    # Use MathVerifier for advanced_math domain
    if domain == "advanced_math" and execution_result is not None and final_answer is not None:
        try:
            math_verifier = await get_math_verifier()
            verification_result = await math_verifier.verify(
                solution=execution_result,
                reference_answer=final_answer
            )
            return verification_result.status == VerificationOutcome.SUCCESS
        except Exception:
            pass
    
    if execution_result == final_answer:
        return True


# Cache to store verifiers by package requirements
_verifier_cache = {}

# Initialize math verifier for advanced_math domain
_math_verifier = None

# Initialize physics verifier for advanced_physics domain
_physics_verifier = None


async def get_math_verifier():
    """
    Get or initialize the MathVerifier instance for advanced_math domain.
    
    Returns:
        MathVerifier instance
    """
    global _math_verifier
    if _math_verifier is None:
        _math_verifier = MathVerifier(float_rounding=6, numeric_precision=15)
        await _math_verifier.setup()
    return _math_verifier

async def get_physics_verifier():
    """
    Get or initialize the PhysicsVerifier instance for advanced_physics domain.

    Returns:
        PhysicsVerifier instance
    """
    global _physics_verifier
    if _physics_verifier is None:
        _physics_verifier = PhysicsVerifier(float_rounding=6, numeric_precision=15) 
        await _physics_verifier.setup(uv=True)
    return _physics_verifier

async def get_or_create_verifier(required_packages: List[str]) -> Tuple[PythonVerifier, bool]:
    """
    Get a verifier from cache or create a new one if needed.
    
    Args:
        required_packages: List of required packages with versions.
        
    Returns:
        Tuple of (verifier, is_from_cache)
    """
    # Sort packages to ensure consistent cache keys
    cache_key = tuple(sorted(required_packages))
    
    if ENV_CACHE_ENABLED and cache_key in _verifier_cache:
        return _verifier_cache[cache_key], True
    
    # Create a new verifier
    verifier = await setup_verifier(required_packages)
    
    if ENV_CACHE_ENABLED:
        _verifier_cache[cache_key] = verifier
    
    return verifier, False

async def process_batch(batch: List[Tuple[int, Dict[str, Any]]], required_packages: List[str], domain: str = None) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Process a batch of items using a single verifier.
    
    Args:
        batch: List of (index, item) tuples to process.
        required_packages: List of required packages with versions.
        
    Returns:
        List of (index, result) tuples.
    """
    if not batch:
        return []
    
    # Get or create a verifier for this batch
    verifier, from_cache = await get_or_create_verifier(required_packages)
    
    try:
        # Process items concurrently within the batch
        tasks = [process_single_item(item_tuple, verifier, domain) for item_tuple in batch]
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []
    finally:
        # Only clean up if not from cache and caching is disabled
        if not ENV_CACHE_ENABLED and not from_cache and verifier.venv_path:
            await verifier.cleanup()


async def process_single_item(item_tuple: Tuple[int, Dict[str, Any]], verifier: PythonVerifier, domain: str = None) -> Tuple[int, Dict[str, Any]]:
    """
    Process a single item using an existing verifier.
    
    Args:
        item_tuple: Tuple containing (index, item) from the dataset.
        verifier: The Python verifier to use.
        
    Returns:
        Tuple of (index, result dictionary).
    """
    idx, item = item_tuple
    
    rationale = item.get("rationale", "")
    final_answer = item.get("final_answer", "")

    # Handle advanced_physics domain using PhysicsVerifier directly
    if domain == "advanced_physics":
        # Check if rationale and final_answer exist
        if not rationale or not final_answer:
            logger.warning(f"Skipping item {idx} due to missing rationale or final_answer.")
            # Ensure this error case returns the expected keys
            return idx, {
                "execution_status": "MISSING_DATA",
                "execution_result": "",
                "execution_successful": False,
                "result": "",
                "match_status": False,
                "error_message": "Missing rationale or final_answer for Physics verification."
            }
        try:
            physics_verifier = await get_physics_verifier()
            verification_result = await physics_verifier.verify(
                solution=rationale,
                reference_answer=final_answer
            )
            # Map the verifier output to the expected dictionary structure
            return idx, {
                "execution_status": verification_result.status.name, # Map from status
                "execution_result": verification_result.result,     # Map from result
                "status": verification_result.status.name, # Keep original keys too
                "result": verification_result.result,
                "error_message": verification_result.error_message,
                "execution_successful": verification_result.status == VerificationOutcome.SUCCESS,
                "match_status": verification_result.status == VerificationOutcome.SUCCESS
            }
        except Exception as e:
            logger.error(f"Error during physics verification for item {idx}: {e}")
            return idx, {
                "execution_status": "VERIFIER_ERROR",
                "execution_result": "",
                "status": "ERROR",
                "result": "",
                "error_message": f"Physics verification error: {str(e)}",
                "execution_successful": False,
                "match_status": False
            }

    # Existing logic for other domains
    if not rationale or not final_answer:
        return idx, {
            "execution_status": "MISSING_DATA",
            "execution_result": "",
            "execution_successful": False,
            "result": "",
            "match_status": False,
            "error_message": "Missing rationale or final_answer."
        }

    # Execute rationale for other domains
    execution_output = await execute_rationale(rationale, verifier)
    # Add execution_status here as well
    execution_status = "SUCCESS" if execution_output["execution_successful"] else "FAILURE"

    # Compare results for other domains
    if execution_output["execution_successful"]:
        match_status = await compare_results(execution_output["result"], final_answer, domain)
        return idx, {
            "execution_status": execution_status,
            "execution_successful": True,
            "execution_result": execution_output["result"],
            "match_status": match_status,
            "result": execution_output["result"],
            "error_message": execution_output.get("error_message", "")
        }
    else:
        # Execution failed for other domains
        return idx, {
            "execution_status": execution_status,
            "execution_successful": False,
            "execution_result": "",
            "match_status": False,
            "result": "",
            "error_message": execution_output.get("error_message", "Execution failed")
        }


async def group_by_packages(items: List[Tuple[int, Dict[str, Any]]]) -> Dict[Tuple[str, ...], List[Tuple[int, Dict[str, Any]]]]:  
    """
    Group items by their required packages for more efficient processing.
    
    Args:
        items: List of (index, item) tuples.
        
    Returns:
        Dictionary mapping package combinations to lists of items.
    """
    grouped = {}
    
    for item_tuple in items:
        _, item = item_tuple
        # Check multiple possible locations for dependencies
        packages = []
        
        # Check in meta_data.required_dependencies (physics domain)
        if item.get("meta_data", {}).get("required_dependencies"):
            packages = item.get("meta_data", {}).get("required_dependencies", [])
        # Check in metadata.required_packages (finance domain)
        elif item.get("metadata", {}).get("required_dependencies"):
            packages = item.get("metadata", {}).get("required_dependencies", [])
        # Check for Mathematical Programming domain that needs pyscipopt
        elif item.get("metadata", {}).get("domain") == "Mathematical_Programming" or \
             (item.get("metadata", {}).get("library") == "SCIP"):
            packages = ["pyscipopt", "pandas", "gurobipy", "cvxpy", "matplotlib",]
        
        # 确保medcalc-bench被添加到所有包组中
        if "medcalc-bench" not in packages:
            packages.append("medcalc-bench")
        
        # Sort packages to ensure consistent grouping
        key = tuple(sorted(packages))
        
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item_tuple)
    
    return grouped

def generate_detailed_domain_files(dataset: Dict[str, List[Dict[str, Any]]], results: Dict[str, List[Dict[str, Any]]], output_dir: str) -> None:
    """
    Generate detailed JSON files for each domain, including all original information
    from the seed dataset along with execution status, results, and matching information.
    
    Args:
        dataset: Original dataset dictionary with items grouped by domain.
        results: Dictionary with execution results for each domain.
        output_dir: Directory to save the detailed JSON files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for domain, domain_results in results.items():
        # Get the original items for this domain
        domain_items = dataset.get(domain, [])
        
        # Combine original data with execution results
        detailed_items = []
        for i, (item, result) in enumerate(zip(domain_items, domain_results)):
            # Create a copy of the original item
            detailed_item = item.copy()
            
            # Add execution results
            detailed_item["execution"] = {
                "status": result["execution_status"],
                "successful": result["execution_successful"],
                "match_status": result["match_status"]
            }
            
            # Add execution result or error message
            if result["execution_successful"]:
                detailed_item["execution"]["result"] = result["execution_result"]
            else:
                detailed_item["execution"]["error_message"] = result.get("error_message", "Unknown error")
            
            detailed_items.append(detailed_item)
        
        # Save detailed items to a JSON file for this domain
        domain_file_path = os.path.join(output_dir, f"{domain}_detailed.json")
        with open(domain_file_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_items, f, indent=2, ensure_ascii=False)
        
        print(f"Generated detailed JSON file for domain '{domain}': {domain_file_path}")

async def process_dataset(
    dataset_path: str, 
    output_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_samples: Optional[int] = None,
    concurrent_batches: int = DEFAULT_CONCURRENT_BATCHES,
    detailed_output_dir: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a dataset file and save results using batch processing.
    
    Args:
        dataset_path: Path to the dataset JSON file.
        output_path: Optional path to save results.
        batch_size: Number of items to process in each batch.
        max_samples: Optional maximum number of samples to process per domain.
        concurrent_batches: Number of batches to process concurrently.
        detailed_output_dir: Optional directory to save detailed JSON files for each domain.
        
    Returns:
        Dictionary with processing results for each domain.
    """
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Make a copy of the original dataset for later use
    original_dataset = {}
    results = {}
    total_start_time = time.time()
    total_items_processed = 0
    
    # 检查 dataset 的类型，处理列表或字典类型
    if isinstance(dataset, list):
        # 如果 dataset 是列表，则所有项都放在默认域名下处理
        print(f"Processing dataset as a list with {len(dataset)} items...")
        domain = "default"
        items = dataset
        
        # 将列表转换为单个域的字典结构
        dataset = {"default": items}
        original_dataset = {"default": items.copy()}
    else:
        # 对于字典类型的 dataset，保持原来的处理方式
        print(f"Processing {len(dataset)} domains...")
    
    # Process each domain
    for domain, items in dataset.items():
        domain_start_time = time.time()
        
        # Limit the number of samples if specified
        if max_samples is not None:
            items = items[:max_samples]
        
        # Store the original items for this domain if not already stored
        if domain not in original_dataset:
            original_dataset[domain] = items.copy()
        
        print(f"\nProcessing domain: {domain} ({len(items)} items)")
        
        # Create indexed items
        indexed_items = list(enumerate(items))
        
        # Group items by their required packages
        grouped_items = await group_by_packages(indexed_items)
        print(f"Grouped into {len(grouped_items)} distinct package configurations")
        
        # Process batches with progress bar
        all_results = []
        with tqdm(total=len(items), desc=f"Domain: {domain}") as pbar:
            # Process each package group
            for packages, group_items in grouped_items.items():
                # Create batches within this package group
                group_batches = [group_items[i:i+batch_size] for i in range(0, len(group_items), batch_size)]
                
                # Process batches concurrently in groups
                for i in range(0, len(group_batches), concurrent_batches):
                    batch_group = group_batches[i:i+concurrent_batches]
                    
                    # Process this group of batches concurrently
                    batch_tasks = [process_batch(batch, list(packages), domain) for batch in batch_group]
                    group_results = await asyncio.gather(*batch_tasks)
                    
                    # Flatten the results
                    for batch_result in group_results:
                        all_results.extend(batch_result)
                        pbar.update(len(batch_result))
        
        # Sort results by index and extract just the result dictionaries
        sorted_results = [result for _, result in sorted(all_results, key=lambda x: x[0])]
        results[domain] = sorted_results
        
        # Domain statistics
        domain_time = time.time() - domain_start_time
        successful_executions = sum(1 for r in sorted_results if r["execution_successful"])
        successful_matches = sum(1 for r in sorted_results if r["match_status"])
        
        print(f"Domain: {domain} - Completed in {domain_time:.2f} seconds")
        print(f"  Successful executions: {successful_executions}/{len(sorted_results)} ({successful_executions/len(sorted_results)*100:.2f}%)")
        print(f"  Successful matches: {successful_matches}/{len(sorted_results)} ({successful_matches/len(sorted_results)*100:.2f}%)")
        
        total_items_processed += len(sorted_results)
    
    # Save results if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate detailed JSON files for each domain if output directory is provided
    if detailed_output_dir:
        generate_detailed_domain_files(original_dataset, results, detailed_output_dir)
    
    # Overall statistics
    total_time = time.time() - total_start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per item: {total_time/total_items_processed:.2f} seconds")
    
    # Clean up verifier cache at the end
    if ENV_CACHE_ENABLED and _verifier_cache:
        print(f"Cleaning up {len(_verifier_cache)} cached verifiers...")
        cleanup_tasks = [verifier.cleanup() for verifier in _verifier_cache.values() if verifier.venv_path]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks)
        _verifier_cache.clear()
    
    return results


def generate_visualizations(results: Dict[str, List[Dict[str, Any]]], output_dir: str) -> None:
    """
    Generate visualizations of the execution results.
    
    Args:
        results: Dictionary with execution results for each domain.
        output_dir: Directory to save visualizations.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    domain_stats = []
    for domain, items in results.items():
        total = len(items)
        successful_executions = sum(1 for item in items if item["execution_successful"])
        successful_matches = sum(1 for item in items if item["match_status"])
        
        domain_stats.append({
            "Domain": domain,
            "Total Samples": total,
            "Successful Executions": successful_executions,
            "Successful Matches": successful_matches,
            "Execution Success Rate": (successful_executions / total * 100) if total > 0 else 0,
            "Match Rate": (successful_matches / total * 100) if total > 0 else 0
        })
    
    # Convert to DataFrame and sort by match rate
    df = pd.DataFrame(domain_stats)
    df = df.sort_values("Match Rate", ascending=False)
    
    # 1. Bar chart of execution success rate and match rate by domain
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    x = np.arange(len(df))
    
    plt.bar(x - bar_width/2, df["Execution Success Rate"], bar_width, label="Execution Success Rate (%)")
    plt.bar(x + bar_width/2, df["Match Rate"], bar_width, label="Match Rate (%)")
    
    plt.xlabel("Domain")
    plt.ylabel("Percentage")
    plt.title("Execution Success Rate and Match Rate by Domain")
    plt.xticks(x, df["Domain"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_and_match_rates.png"))
    plt.close()
    
    # 2. Stacked bar chart of execution and match counts by domain
    plt.figure(figsize=(12, 8))
    
    # Calculate failed executions and failed matches for stacked bar
    df["Failed Executions"] = df["Total Samples"] - df["Successful Executions"]
    df["Failed Matches"] = df["Successful Executions"] - df["Successful Matches"]
    
    # Execution success/failure
    plt.subplot(2, 1, 1)
    plt.bar(df["Domain"], df["Successful Executions"], label="Successful Executions", color="green")
    plt.bar(df["Domain"], df["Failed Executions"], bottom=df["Successful Executions"], label="Failed Executions", color="red")
    plt.xlabel("Domain")
    plt.ylabel("Number of Samples")
    plt.title("Execution Results by Domain")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    
    # Match success/failure (only for successful executions)
    plt.subplot(2, 1, 2)
    plt.bar(df["Domain"], df["Successful Matches"], label="Successful Matches", color="blue")
    plt.bar(df["Domain"], df["Failed Matches"], bottom=df["Successful Matches"], label="Failed Matches", color="orange")
    plt.xlabel("Domain")
    plt.ylabel("Number of Samples")
    plt.title("Match Results by Domain (Only Successful Executions)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "execution_and_match_counts.png"))
    plt.close()
    
    # 3. Save summary table as CSV
    summary_df = df[["Domain", "Total Samples", "Execution Success Rate", "Match Rate"]]
    summary_df.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)
    
    # 4. Create a summary markdown file
    with open(os.path.join(output_dir, "summary_results.md"), "w") as f:
        f.write("# Execution and Comparison Results Summary\n\n")
        f.write("## Overall Statistics\n\n")
        
        total_samples = df["Total Samples"].sum()
        avg_execution_rate = df["Execution Success Rate"].mean()
        avg_match_rate = df["Match Rate"].mean()
        
        f.write(f"- **Total Samples Analyzed**: {total_samples}\n")
        f.write(f"- **Average Execution Success Rate**: {avg_execution_rate:.2f}%\n")
        f.write(f"- **Average Match Rate**: {avg_match_rate:.2f}%\n\n")
        
        f.write("## Domain-Specific Results\n\n")
        f.write(summary_df.to_markdown(index=False))


async def main():
    """
    Main function to execute and compare rationale code with final answers.
    """
    import argparse
    import subprocess
    
    # 首先尝试安装 medcalc-bench 包
    print("尝试安装 medcalc-bench 包...")
    try:
        subprocess.check_call(["pip", "install", "medcalc-bench"])
        print("medcalc-bench 安装成功")
    except Exception as e:
        print(f"安装 medcalc-bench 时出错: {e}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default paths relative to the script location
    default_dataset = os.path.join(script_dir, "seed_dataset_all_domain.json")
    default_output = os.path.join(script_dir, "execution_comparison_results.json")
    default_vis_dir = os.path.join(script_dir, "execution_comparison_visualizations")
    default_detailed_dir = os.path.join(script_dir, "detailed_domain_results")
    
    parser = argparse.ArgumentParser(description="Execute rationale code and compare with final answers")
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        help="Path to the dataset JSON file")
    parser.add_argument("--output", type=str, default=default_output,
                        help="Path to save the results")
    parser.add_argument("--vis-dir", type=str, default=default_vis_dir,
                        help="Directory to save visualizations")
    parser.add_argument("--detailed-dir", type=str, default=default_detailed_dir,
                        help="Directory to save detailed JSON files for each domain")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of items to process in each batch")
    parser.add_argument("--concurrent-batches", type=int, default=DEFAULT_CONCURRENT_BATCHES,
                        help="Number of batches to process concurrently")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process per domain")
    parser.add_argument("--single-item", action="store_true",
                        help="Process only a single item (for testing)")
    parser.add_argument("--skip-execution", action="store_true",
                        help="Skip execution and just generate visualizations from existing results")
    parser.add_argument("--disable-cache", action="store_true",
                        help="Disable caching of virtual environments")
    
    args = parser.parse_args()
    
    # Set global configuration
    global ENV_CACHE_ENABLED
    ENV_CACHE_ENABLED = not args.disable_cache
    
    if args.single_item:
        # For testing with a single item
        test_item = {
            "rationale": "import sympy as sp\n\nx, y = sp.symbols('x y', positive=True, real=True)\nresult = -2*(1009)**2\nresult\nprint(result)",
            "final_answer": "-2036162",
            "metadata": {
                "required_dependencies": ["sympy==1.13.3"]
            }
        }
        
        # Use the get_or_create_verifier function to test caching
        verifier, _ = await get_or_create_verifier(["sympy==1.13.3"])
        try:
            _, result = await process_single_item((0, test_item), verifier)
            print(json.dumps(result, indent=2))
        finally:
            if not ENV_CACHE_ENABLED and verifier.venv_path:
                await verifier.cleanup()
    else:
        if args.skip_execution and os.path.exists(args.output):
            # Load existing results
            print(f"Loading existing results from {args.output}")
            with open(args.output, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # If detailed output is requested, we need to load the original dataset
            if args.detailed_dir:
                print(f"Loading original dataset from {args.dataset} for detailed output")
                with open(args.dataset, 'r', encoding='utf-8') as f:
                    original_dataset = json.load(f)
                
                # Generate detailed JSON files
                generate_detailed_domain_files(original_dataset, results, args.detailed_dir)
        else:
            # Process the entire dataset
            print(f"Processing dataset from {args.dataset}")
            results = await process_dataset(
                dataset_path=args.dataset, 
                output_path=args.output,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                concurrent_batches=args.concurrent_batches,
                detailed_output_dir=args.detailed_dir
            )
        
        # Print summary
        total_items = sum(len(items) for items in results.values())
        successful_executions = sum(
            sum(1 for item in items if item["execution_successful"]) 
            for items in results.values()
        )
        successful_matches = sum(
            sum(1 for item in items if item["match_status"]) 
            for items in results.values()
        )
        
        print(f"\nSummary:")
        print(f"Processed {total_items} items across {len(results)} domains")
        print(f"Successful executions: {successful_executions} ({successful_executions/total_items*100:.2f}%)")
        print(f"Successful matches: {successful_matches} ({successful_matches/total_items*100:.2f}%)")
        
        # Generate visualizations
        print(f"\nGenerating visualizations in {args.vis_dir}")
        generate_visualizations(results, args.vis_dir)

    # Ensure verifiers are cleaned up
    print("\nCleaning up verifiers...")
    # Cleanup cached PythonVerifiers if caching is disabled
    if not ENV_CACHE_ENABLED:
        for verifier in _verifier_cache.values():
            if verifier.venv_path:
                await verifier.cleanup()
    
    # Cleanup MathVerifier if initialized
    if _math_verifier:
        await _math_verifier.cleanup()

    # Cleanup PhysicsVerifier if initialized
    if _physics_verifier:
        await _physics_verifier.cleanup()

    print("Processing complete.")


if __name__ == "__main__":
    asyncio.run(main())