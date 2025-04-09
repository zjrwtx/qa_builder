#!/usr/bin/env python3
"""
Compare execution results from the Loong dataset with the original final answers.
This script analyzes the accuracy of code execution results across all domains.
"""

import json
import os
import glob
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datasets import load_dataset
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def load_execution_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load execution results from a JSON file.
    
    Args:
        file_path: Path to the execution results JSON file.
        
    Returns:
        List of execution result dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# def extract_domain_from_filename(file_path: str) -> str:
    """
    Extract domain name from the execution results filename.
    
    Args:
        file_path: Path to the execution results file.
        
    Returns:
        Domain name.
    """
    # Extract the filename without path and extension
    filename = os.path.basename(file_path)
    # Remove the prefix and suffix to get the domain
    match = re.match(r'loong_execution_results_(.+)\.json', filename)
    if match:
        return match.group(1)
    return "unknown"


def normalize_value(value: Any) -> str:
    """
    Normalize a value for comparison by removing whitespace and converting to string.
    Also extracts actual result from execution output that may contain log messages.
    
    Args:
        value: Value to normalize.
        
    Returns:
        Normalized string value.
    """
    if value is None:
        return ""
    
    # Convert to string and strip whitespace
    value_str = str(value).strip()
    
    # Check if the value contains a list representation at the end (common in execution outputs)
    list_match = re.search(r'\[\(.*\)\]$', value_str)
    if list_match:
        # Extract just the list part
        value_str = list_match.group(0)
    
    # Check if the value contains a dictionary representation at the end
    dict_match = re.search(r'\{[^\{\}]*\}$', value_str)
    if dict_match:
        # Extract just the dictionary part
        value_str = dict_match.group(0)
    
    # Try to convert to float for numeric comparison
    try:
        # If it's a numeric value, format it consistently
        float_val = float(value_str)
        # Use scientific notation for very small or large numbers
        if abs(float_val) < 0.0001 or abs(float_val) > 10000:
            return f"{float_val:.10e}"
        # For numbers between 0 and 1, use 5 decimal places
        if 0 <= float_val <= 1:
            return f"{float_val:.5f}".rstrip('0').rstrip('.')
        # Otherwise use fixed precision
        return f"{float_val:.10f}".rstrip('0').rstrip('.') if '.' in f"{float_val:.10f}" else f"{float_val:.10f}"
    except (ValueError, TypeError):
        # For non-numeric values, return as is
        return value_str


def is_numeric_match(result: str, answer: str, tolerance: float = 1e-6) -> bool:
    """
    Check if two numeric values match within a tolerance.
    
    Args:
        result: The execution result.
        answer: The expected answer.
        tolerance: The relative tolerance for floating point comparison.
        
    Returns:
        True if the values match within tolerance, False otherwise.
    """
    try:
        result_float = float(result)
        answer_float = float(answer)
        
        # For very small values, use absolute tolerance
        if abs(answer_float) < 1e-10:
            return abs(result_float - answer_float) < tolerance
        
        # For numbers between 0 and 1, use a slightly larger tolerance
        if 0 <= answer_float <= 1:
            tolerance = max(tolerance, 1e-5)
        
        # Otherwise use relative tolerance
        relative_diff = abs((result_float - answer_float) / answer_float)
        return relative_diff < tolerance
    except (ValueError, TypeError):
        # If conversion to float fails, they're not numeric values
        return False


def values_match(result: str, answer: str) -> Tuple[bool, str]:
    """
    Check if execution result matches the expected answer.
    
    Args:
        result: The execution result.
        answer: The expected answer.
        
    Returns:
        Tuple of (match_status, match_type)
    """
    # Normalize both values for comparison
    norm_result = normalize_value(result)
    norm_answer = normalize_value(answer)
    
    # Exact string match
    if norm_result == norm_answer:
        return True, "exact"
    
    # Numeric match within tolerance
    if is_numeric_match(norm_result, norm_answer):
        return True, "numeric"
    
    # No match
    return False, "none"


def compare_results(execution_results: List[Dict[str, Any]], 
                    dataset_samples: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare execution results with original dataset answers.
    
    Args:
        execution_results: List of execution result dictionaries.
        dataset_samples: Dictionary mapping sample IDs to dataset samples.
        
    Returns:
        Dictionary with comparison statistics.
    """
    total_samples = len(execution_results)
    successful_executions = 0
    correct_answers = 0
    match_types = {"exact": 0, "numeric": 0, "none": 0}
    
    # Detailed results for each sample
    detailed_results = []
    
    for result in execution_results:
        sample_id = result.get("id")
        
        # Skip if sample not found in dataset
        if sample_id not in dataset_samples:
            continue
        
        dataset_sample = dataset_samples[sample_id]
        execution_successful = result.get("execution_result", {}).get("execution_successful", False)
        
        if execution_successful:
            successful_executions += 1
            
            # Get execution result and expected answer
            execution_value = result.get("execution_result", {}).get("result", "")
            expected_answer = dataset_sample.get("final_answer", "")
            
            # Check if values match
            is_match, match_type = values_match(execution_value, expected_answer)
            
            if is_match:
                correct_answers += 1
                match_types[match_type] += 1
            else:
                match_types["none"] += 1
                
            # Record detailed result
            detailed_results.append({
                "id": sample_id,
                "question": result.get("question", ""),
                "execution_successful": execution_successful,
                "execution_value": execution_value,
                "expected_answer": expected_answer,
                "is_match": is_match,
                "match_type": match_type
            })
        else:
            # Record failed execution
            detailed_results.append({
                "id": sample_id,
                "question": result.get("question", ""),
                "execution_successful": execution_successful,
                "execution_value": "",
                "expected_answer": dataset_sample.get("final_answer", ""),
                "is_match": False,
                "match_type": "none",
                "error_message": result.get("execution_result", {}).get("error_message", "")
            })
    
    # Calculate statistics
    accuracy = correct_answers / total_samples if total_samples > 0 else 0
    execution_success_rate = successful_executions / total_samples if total_samples > 0 else 0
    
    return {
        "total_samples": total_samples,
        "successful_executions": successful_executions,
        "correct_answers": correct_answers,
        "execution_success_rate": execution_success_rate,
        "accuracy": accuracy,
        "match_types": match_types,
        "detailed_results": detailed_results
    }


def save_matched_results(comparison_results: Dict[str, Any], dataset_samples: Dict[int, Dict[str, Any]], output_file: str) -> None:
    """
    Save the correctly matched results to a new JSON file in the original data format.
    
    Args:
        comparison_results: Dictionary with comparison results.
        dataset_samples: Dictionary mapping sample IDs to dataset samples.
        output_file: Path to the output JSON file.
    """
    matched_samples = []
    
    for result in comparison_results.get("detailed_results", []):
        if result.get("is_match"):
            # Get the original sample
            sample_id = result.get("id")
            if sample_id in dataset_samples:
                original_sample = dataset_samples[sample_id]
                # Create a sample in the original format
                matched_sample = {
                    "question": original_sample.get("question", ""),
                    "final_answer": original_sample.get("final_answer", ""),
                    "rationale": original_sample.get("rationale", "")
                }
                matched_samples.append(matched_sample)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(matched_samples)} matched samples to {output_file}")


def analyze_domain(domain:str, 
                   execution_file: str, 
                   dataset_name: str = "zjrwtxtechstudio/test2025") -> Dict[str, Any]:
    """
    Analyze results for a specific domain.
    
    Args:
        domain: Domain name.
        execution_file: Path to execution results file.
        dataset_name: Name of the Hugging Face dataset.
        
    Returns:
        Dictionary with analysis results and dataset samples.
    """
    print(f"Analyzing domain: {domain}")
    
    # Load execution results
    execution_results = load_execution_results(execution_file)
    
    # Load dataset samples for this domain
    try:
        dataset = load_dataset(dataset_name, split=domain)
        # Create a dictionary mapping sample IDs to dataset samples
        dataset_samples = {i: sample for i, sample in enumerate(dataset)}
    except Exception as e:
        print(f"Error loading dataset for domain {domain}: {e}")
        # Try loading the dataset with the domain as a filter
        try:
            dataset = load_dataset(dataset_name)
            # Find the appropriate split
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                # Check if this split contains the domain
                if any(sample.get("source_type") == domain for sample in split_data):
                    # Create a dictionary mapping sample IDs to dataset samples
                    dataset_samples = {i: sample for i, sample in enumerate(split_data) 
                                      if sample.get("source_type") == domain}
                    break
            else:
                print(f"Could not find domain {domain} in any split")
                return None
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    # Compare results
    comparison_results = compare_results(execution_results, dataset_samples)
    comparison_results["domain"] = domain
    
    # Print summary
    print(f"  Total samples: {comparison_results['total_samples']}")
    print(f"  Successful executions: {comparison_results['successful_executions']} "
          f"({comparison_results['execution_success_rate']*100:.2f}%)")
    print(f"  Correct answers: {comparison_results['correct_answers']} "
          f"({comparison_results['accuracy']*100:.2f}%)")
    print(f"  Match types: {comparison_results['match_types']}")
    
    return comparison_results, dataset_samples


def generate_visualizations(all_results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Generate visualizations of the comparison results.
    
    Args:
        all_results: List of domain analysis results.
        output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame([
        {
            "Domain": r["domain"],
            "Total Samples": r["total_samples"],
            "Execution Success Rate": r["execution_success_rate"] * 100,
            "Accuracy": r["accuracy"] * 100,
            "Exact Matches": r["match_types"]["exact"],
            "Numeric Matches": r["match_types"]["numeric"],
            "No Matches": r["match_types"]["none"]
        }
        for r in all_results if r is not None
    ])
    
    # Sort by accuracy
    df = df.sort_values("Accuracy", ascending=False)
    
    # 1. Bar chart of execution success rate and accuracy by domain
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    x = np.arange(len(df))
    
    plt.bar(x - bar_width/2, df["Execution Success Rate"], bar_width, label="Execution Success Rate (%)")
    plt.bar(x + bar_width/2, df["Accuracy"], bar_width, label="Accuracy (%)")
    
    plt.xlabel("Domain")
    plt.ylabel("Percentage")
    plt.title("Execution Success Rate and Accuracy by Domain")
    plt.xticks(x, df["Domain"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_and_accuracy.png"))
    plt.close()
    
    # 2. Stacked bar chart of match types by domain
    plt.figure(figsize=(12, 8))
    
    bottom = np.zeros(len(df))
    for match_type, color in [("Exact Matches", "green"), ("Numeric Matches", "blue"), ("No Matches", "red")]:
        plt.bar(df["Domain"], df[match_type], bottom=bottom, label=match_type, color=color)
        bottom += df[match_type]
    
    plt.xlabel("Domain")
    plt.ylabel("Number of Samples")
    plt.title("Match Types by Domain")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "match_types.png"))
    plt.close()
    
    # 3. Save summary table as CSV
    df.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)
    
    # 4. Create a summary markdown file
    with open(os.path.join(output_dir, "summary_results.md"), "w") as f:
        f.write("# Loong Dataset Execution Results Summary\n\n")
        f.write("## Overall Statistics\n\n")
        
        total_samples = df["Total Samples"].sum()
        avg_success_rate = df["Execution Success Rate"].mean()
        avg_accuracy = df["Accuracy"].mean()
        
        f.write(f"- **Total Samples Analyzed**: {total_samples}\n")
        f.write(f"- **Average Execution Success Rate**: {avg_success_rate:.2f}%\n")
        f.write(f"- **Average Accuracy**: {avg_accuracy:.2f}%\n\n")
        
        f.write("## Domain-Specific Results\n\n")
        f.write(df.to_markdown(index=False))


def main():
    """
    Main function to compare execution results with original dataset answers.
    """
    # Base directory for types
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # base_dir = "./loong_results"
    base_dir = "."
    # Find all execution result files
    execution_files = glob.glob(os.path.join(base_dir, "loong_execution_results_*.json"))
    
    # Output directory for visualizations
    output_dir = os.path.join(base_dir, "comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each domain
    all_results = []
    for execution_file in execution_files:
        # domain = extract_domain_from_filename(execution_file)
        result_data = analyze_domain("train", execution_file)
        if result_data:
            comparison_results, dataset_samples = result_data
            all_results.append(comparison_results)
            
            # Save matched results in the original format
            matched_output_file = os.path.join(output_dir, f"matched_samples_train.json")
            save_matched_results(comparison_results, dataset_samples, matched_output_file)
            
            # Save detailed results for this domain
            detailed_file = os.path.join(output_dir, f"detailed_train.json")
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_results["detailed_results"], f, indent=2, ensure_ascii=False)
    
    # Generate visualizations
    generate_visualizations(all_results, output_dir)
    
    # Save overall results
    overall_file = os.path.join(output_dir, "overall_results.json")
    with open(overall_file, 'w', encoding='utf-8') as f:
        # Remove detailed results to keep the file size manageable
        for result in all_results:
            if "detailed_results" in result:
                del result["detailed_results"]
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()