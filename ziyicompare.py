#!/usr/bin/env python3
"""
Script to combine seed datasets from all domains into a single JSON file.
This script reads seed_dataset.json files from each domain directory
and combines them into a single JSON file named 'seed_dataset_all_domain.json'.
"""

import json
from pathlib import Path
import logging
import asyncio
import json
import time
import ast
import math


from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import sys

from camel.verifiers.python_verifier import PythonVerifier
from camel.verifiers.models import VerificationOutcome
from camel.logger import disable_logging
disable_logging()
# from math_verifier_tem import MathVerifier
from physic_verifier_tem import PhysicsVerifier
from camel.extractors import BaseExtractor, BoxedStrategy

# Configuration constants
DEFAULT_MAX_WORKERS = 6
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 3600.0
DEFAULT_CONCURRENT_BATCHES = 10  # Number of batches to process concurrently
ENV_CACHE_ENABLED = True        # Enable caching of virtual environments

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)


def combine_seed_data():
    # Define the base data directory
    data_dir = Path(__file__).parent

    # Define the domains to process with their paths
    domain_paths = {
        "advanced_physics": data_dir / "advanced_physics" / "seed_dataset.json",
        "computational_biology": data_dir / "computational_biology" / "seed_dataset.json",
        "finance": data_dir / "finance" / "seed_dataset.json",
        "games": data_dir / "games" / "blackjack" / "seed_dataset.json",  # Special case for games
        "graph_discrete_math": data_dir / "graph_discrete_math" / "seed_dataset.json",
        "logic": data_dir / "logic" / "seed_dataset.json",
        "mathematical_programming": data_dir / "mathematical_programming" / "seed_dataset.json",
        "security_and_safety": data_dir / "security_and_safety" / "seed_dataset.json",
        "advanced_math": data_dir / "advanced_math" / "seed_dataset.json",
    }
    # Dictionary to hold all domain data
    all_domains_data = {}

    # Process each domain
    for domain, domain_path in domain_paths.items():
        if domain_path.exists():
            logger.info(f"Processing {domain}...")
            try:
                with open(domain_path, 'r', encoding='utf-8') as f:
                    domain_data = json.load(f)
                    all_domains_data[domain] = domain_data
                logger.info(f"Successfully loaded {domain} with {len(domain_data)} entries")
            except Exception as e:
                logger.error(f"Error loading {domain}: {e}")
        else:
            logger.warning(f"Warning: {domain_path} does not exist")
    
    # Write the combined data to a new file
    output_file = data_dir / "seed_dataset_all_domain.json"
    
    logger.info(f"\nWriting combined data to {output_file}...")
    
    # Print summary
    total_domains = len(all_domains_data)
    logger.info(f"\nCombined {total_domains} domains into {output_file}")
    logger.info("Domains included:")
    for domain in all_domains_data:
        logger.info(f"- {domain}: {len(all_domains_data[domain])} entries")

    return all_domains_data


async def setup_verifier(required_packages: List[str], timeout: float = DEFAULT_TIMEOUT, domain: str = None) -> PythonVerifier:
    """
    Set up a Python verifier with the required packages.
    
    Args:
        required_packages: List of required packages with versions.
        timeout: Timeout for code execution in seconds.
        domain: The problem domain (e.g., 'mathematical_programming').
        
    Returns:
        A configured PythonVerifier instance.
    """
    # Set longer timeout for Mathematical Programming domain
    if domain == "mathematical_programming":
        timeout = 1200.0
    required_packages=required_packages+["medcalc-bench"]
    verifier = PythonVerifier(
        timeout=timeout, 
        required_packages=required_packages)
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

def parse_answer_str(s):
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Failed to parse string: {s}\nError: {e}")


def extract_numbers(data):
    numbers = []

    def recurse(x, path=""):
        if isinstance(x, (int, float)):
            numbers.append((path, x))
        elif isinstance(x, (list, tuple)):
            for i, item in enumerate(x):
                recurse(item, f"{path}[{i}]")
        elif isinstance(x, dict):
            for k, v in x.items():
                recurse(v, f"{path}.{k}" if path else k)

    recurse(data)
    return [num for _, num in sorted(numbers)]


def compare_answer_str(s1, s2, rel_tol=0, abs_tol=0):
    d1 = parse_answer_str(s1)
    d2 = parse_answer_str(s2)
    nums1 = extract_numbers(d1)
    nums2 = extract_numbers(d2)

    if len(nums1) != len(nums2):
        return False, list(zip(nums1, nums2))

    results = [math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(nums1, nums2)]
    return all(results)

async def compare_results(
        execution_result: str, 
        final_answer: str, 
        domain: str = None,
        precision: float = None) -> bool:
    """
    Enhanced comparison between execution result and final answer.
    Performs normalization before comparison for more accurate matching.
    For advanced_math domain, uses MathVerifier for more sophisticated comparison.
    
    Args:
        execution_result: The result from code execution.
        final_answer: The expected final answer.
        domain: The problem domain (e.g., 'advanced_math').
        precision: The precision of the final answer.
        
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
    elif domain == "mathematical_programming" and execution_result is not None and final_answer is not None:
        try:
            math_programming_verifier = await get_math_programming_verifier()
            verification_result = await math_programming_verifier.verify(
                solution=execution_result,
                reference_answer=final_answer
            )
            return verification_result.status == VerificationOutcome.SUCCESS
        except Exception:
            pass
    elif domain == "advanced_physics" and execution_result is not None and final_answer is not None:
        try:
            physics_verifier = await get_physics_verifier()
            verification_result = await physics_verifier.verify(
                solution=execution_result,
                reference_answer=final_answer
            )
            return verification_result.status == VerificationOutcome.SUCCESS
        except Exception:
            pass
    elif precision:
        return compare_answer_str(
            execution_result,
            final_answer,
            rel_tol=precision,
            abs_tol=precision)
    elif execution_result == final_answer:
        return True
    elif compare_answer_str(execution_result, final_answer):
        return True
    else:
        return False


# Cache to store verifiers by package requirements
_verifier_cache = {}

# Initialize math verifier for advanced_math domain
_math_verifier = None

# Initialize physics verifier for advanced_physics domain
_physics_verifier = None

# Initialize mathematical programming verifier for mathematical_programming domain
_math_programming_verifier = None

# async def get_math_verifier():
#     """
#     Get or initialize the MathVerifier instance for advanced_math domain.
    
#     Returns:
#         MathVerifier instance
#     """
#     global _math_verifier
#     if _math_verifier is None:
#         _math_verifier = MathVerifier(float_rounding=6, numeric_precision=15)
#         await _math_verifier.setup()
#     return _math_verifier


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


async def get_math_programming_verifier():
    """
    Get or initialize the Mathematical Programming verifier.
    
    Returns:
        Mathematical Programming verifier
    """
    global _math_programming_verifier
    if _math_programming_verifier is None:
        # Initialize extractor
        extractor = BaseExtractor([[BoxedStrategy()]])
        await extractor.setup()
        timeout = 300.0
        _math_programming_verifier = PythonVerifier(
            timeout=timeout, 
            required_packages=["pyscipopt", "pandas", "gurobipy", "cvxpy", "matplotlib", "geopy"],
            extractor=extractor
        )
        await _math_programming_verifier.setup(uv=True)
    return _math_programming_verifier


async def get_or_create_verifier(required_packages: List[str], domain: str = None):
    """
    Get a verifier from cache or create a new one if needed.
    
    Args:
        required_packages: List of required packages with versions.
        domain: The problem domain (e.g., 'mathematical_programming').
        
    Returns:
        Tuple of (verifier, is_from_cache)
    """
    # Use specialized verifiers for specific domains
    if domain == "advanced_math":
        return await get_math_verifier(), False
    elif domain == "advanced_physics":
        return await get_physics_verifier(), False
    # For other domains, use the package-based caching
    key = tuple(sorted(required_packages))
    
    if ENV_CACHE_ENABLED and key in _verifier_cache:
        return _verifier_cache[key], True
    
    verifier = await setup_verifier(required_packages, domain=domain)
    
    if ENV_CACHE_ENABLED:
        _verifier_cache[key] = verifier
    
    return verifier, False


async def process_batch(batch: List[Tuple[int, Dict[str, Any]]], required_packages: List[str], domain: str = None):
    """
    Process a batch of items using a single verifier.
    
    Args:
        batch: List of (index, item) tuples to process.
        required_packages: List of required packages with versions.
        domain: The problem domain (e.g., 'mathematical_programming').
        
    Returns:
        List of (index, result) tuples.
    """
    if not batch:
        return []
        # Get or create a verifier for this batch
    verifier, from_cache = await get_or_create_verifier(required_packages, domain=domain)
    
    try:
        # Process items concurrently within the batch
        tasks = [process_single_item(item_tuple, verifier, domain) for item_tuple in batch]
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
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
        domain: The problem domain (e.g., 'mathematical_programming').
        
    Returns:
        Tuple of (index, result dictionary).
    """
    idx, item = item_tuple
    
    rationale = item.get("rationale", "")
    final_answer = item.get("final_answer", "")
    precision = item.get("metadata", {}).get("answer_tolerance", None)

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
        match_status = await compare_results(execution_output["result"], final_answer, domain, precision)
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

        # Check in metadata.required_packages (all domain)
        if item.get("metadata", {}).get("required_dependencies"):
            packages = item.get("metadata", {}).get("required_dependencies", [])
        # Check for Mathematical Programming domain that needs pyscipopt
        # elif item.get("metadata", {}).get("domain") == "Mathematical Programming" or \
        #      (item.get("metadata", {}).get("name") == "SCIP"):
        #     packages = ["pyscipopt", "pandas", "gurobi", "cvxpy", "matplotlib", "geopy"]

        # Sort packages to ensure consistent grouping
        key = tuple(sorted(packages))

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item_tuple)

    return grouped


def generate_detailed_domain(dataset: Dict[str, List[Dict[str, Any]]], results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Generate detailed JSON files for each domain, including only failed or unmatched items
    from the seed dataset along with execution status, results, and matching information.
    
    Args:
        dataset: Original dataset dictionary with items grouped by domain.
        results: Dictionary with execution results for each domain.
        
    Returns:
        Dictionary containing failed items grouped by domain.
    """
    # Organize failed cases by domain
    failed_items_by_domain = {}
    
    for domain, domain_results in results.items():
        domain_items = dataset.get(domain, [])
        domain_failed_items = []
        
        for i, (item, result) in enumerate(zip(domain_items, domain_results)):
            if not result["execution_successful"] or not result["match_status"]:
                failed_item = {
                    "domain": domain,
                    "id": item.get("id", i),
                    "expected_final_answer": item.get("final_answer", "N/A"),
                }
                
                if result["execution_successful"]:
                    failed_item["execution_result"] = result["execution_result"]
                else:
                    failed_item["error"] = result.get("error_message", "Unknown error")

                domain_failed_items.append(failed_item)
        
        if domain_failed_items:
            failed_items_by_domain[domain] = domain_failed_items
    
    # Print failed cases by domain
    logger.info("\nFailed Cases by Domain:")
    total_failed = 0
    for domain, failed_items in failed_items_by_domain.items():
        logger.info(f"\n{domain} - Failed Cases ({len(failed_items)}):")
        for item in failed_items:
            logger.info(json.dumps(item, indent=2, ensure_ascii=False))
        total_failed += len(failed_items)
    
    logger.info(f"\nTotal failed cases across all domains: {total_failed}")
    return failed_items_by_domain

async def process_dataset(
    dataset: Dict, 
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_samples: Optional[int] = None,
    concurrent_batches: int = DEFAULT_CONCURRENT_BATCHES,
) -> Tuple[Dict[str, List[Dict[str, Any]]], bool]:
    """
    Process a dataset file and save results using batch processing.
    
    Returns:
        Tuple containing:
        - Dictionary with processing results for each domain
        - Boolean indicating if all domains passed validation (100% success rate)
    """
    original_dataset = {}
    results = {}
    all_domains_passed = True
    
    total_start_time = time.time()
    total_items_processed = 0
    
    # Process each domain
    logger.info(f"Processing {len(dataset)} domains...")
    for domain, items in dataset.items():
        domain_start_time = time.time()
        
        # Limit the number of samples if specified
        if max_samples is not None:
            items = items[:max_samples]
        
        # Store the original items for this domain
        original_dataset[domain] = items.copy()
        
        logger.info(f"\nProcessing domain: {domain} ({len(items)} items)")
        
        # Create indexed items
        indexed_items = list(enumerate(items))
        
        # Group items by their required packages
        grouped_items = await group_by_packages(indexed_items)
        logger.info(f"Grouped into {len(grouped_items)} distinct package configurations")
        
        # Process batches with progress bar
        all_results = []
        with tqdm(total=len(items), desc=f"Domain: {domain}") as pbar:
            # Process each package group
            for batch_idx, (packages, group_items) in enumerate(grouped_items.items()):
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
        
        # Domain statistics and validation
        domain_time = time.time() - domain_start_time
        successful_executions = sum(1 for r in sorted_results if r["execution_successful"])
        successful_matches = sum(1 for r in sorted_results if r["match_status"])
        
        execution_rate = successful_executions/len(sorted_results)*100
        match_rate = successful_matches/len(sorted_results)*100
        
        logger.info(f"Domain: {domain} - Completed in {domain_time:.2f} seconds")
        logger.info(f"  Successful executions: {successful_executions}/{len(sorted_results)} ({execution_rate:.2f}%)")
        logger.info(f"  Successful matches: {successful_matches}/{len(sorted_results)} ({match_rate:.2f}%)")
        
        # Generate detailed domain report for current domain
        logger.info(f"\nFailed cases details for domain {domain}:")
        domain_dataset = {domain: original_dataset[domain]}
        domain_results = {domain: results[domain]}
        failed_cases = generate_detailed_domain(domain_dataset, domain_results)
        
        # Check if all tests passed
        if execution_rate < 100 or match_rate < 100:
            all_domains_passed = False
            logger.error(f"Domain {domain} failed validation!")
            logger.error(f"  Execution rate: {execution_rate:.2f}%")
            logger.error(f"  Match rate: {match_rate:.2f}%")
        
        total_items_processed += len(sorted_results)

    # Overall statistics
    total_time = time.time() - total_start_time
    logger.info(f"\nTotal processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per item: {total_time/total_items_processed:.2f} seconds")
    
    # Clean up verifier cache at the end
    if ENV_CACHE_ENABLED and _verifier_cache:
        logger.info(f"Cleaning up {len(_verifier_cache)} cached verifiers...")
        cleanup_tasks = [verifier.cleanup() for verifier in _verifier_cache.values() if verifier.venv_path]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks)
        _verifier_cache.clear()
    
    return results, all_domains_passed


async def main():
    """
    Main function to execute and compare rationale code with final answers.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Execute rationale code and compare with final answers")

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
    parser.add_argument("--file_path", type=str,
                        help="Path to specific seed_dataset.json file to process")

    args = parser.parse_args()

    # Set global configuration
    global ENV_CACHE_ENABLED
    ENV_CACHE_ENABLED = not args.disable_cache

    # If file_path is provided, only process that specific file
    if args.file_path:
        # Extract domain name from file path
        file_path = Path(args.file_path)
        # Handle both absolute and relative paths
        if not file_path.is_absolute():
            # If path starts with 'data/', remove it as we're already in the data directory
            if str(file_path).startswith('data/'):
                file_path = Path(str(file_path)[5:])
            file_path = Path(__file__).parent / file_path
        
        domain = file_path.parent.name
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
                dataset = {domain: domain_data}
                logger.info(f"Processing single domain {domain} from {file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            logger.error("Please ensure the file path is correct relative to the script location")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            sys.exit(1)
    else:
        # Original behavior - process all domains
        dataset = combine_seed_data()

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
            logger.info(json.dumps(result, indent=2))
        finally:
            if not ENV_CACHE_ENABLED and verifier.venv_path:
                await verifier.cleanup()
    else:
        # Process the entire dataset
        logger.info(f"Processing dataset from all dataset")
        results, all_passed = await process_dataset(
            dataset=dataset,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            concurrent_batches=args.concurrent_batches,
        )

        if not all_passed:
            logger.error("Validation failed! Not all domains achieved 100% success rate.")
            sys.exit(1)  # Make GitHub Action fail
        else:
            logger.info("All domains passed validation with 100% success rate!")

    # Ensure verifiers are cleaned up
    logger.info("\nCleaning up verifiers...")
    # Cleanup cached PythonVerifiers if caching is disabled
    if not ENV_CACHE_ENABLED:
        for verifier in _verifier_cache.values():
            if verifier.venv_path:
                await verifier.cleanup()

    # # Cleanup MathVerifier if initialized
    # if _math_verifier:
    #     await _math_verifier.cleanup()

    # Cleanup PhysicsVerifier if initialized
    if _physics_verifier:
        await _physics_verifier.cleanup()

    logger.info("Processing complete.")


if __name__ == "__main__":
    asyncio.run(main())