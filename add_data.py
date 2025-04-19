#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        sys.exit(1)

def select_records_by_calc_name(data):
    """
    Select one record for each unique calculation name.
    Returns a dictionary with calc_name as keys and selected records as values.
    """
    # Group by calculation name
    grouped_by_calc = defaultdict(list)
    
    # Check if data is a list or needs to be extracted from a wrapper
    records = data
    if not isinstance(data, list) and isinstance(data, dict):
        # Attempt to find a list inside the dictionary
        for key, value in data.items():
            if isinstance(value, list):
                records = value
                break
    
    # Process each record
    for record in records:
        # Skip records without metadata or calc_name
        if 'metadata' not in record or 'calc_name' not in record['metadata']:
            continue
        
        calc_name = record['metadata']['calc_name']
        grouped_by_calc[calc_name].append(record)
    
    # Select one record for each calc_name
    selected_records = {}
    for calc_name, records in grouped_by_calc.items():
        selected_records[calc_name] = records[0]
    
    return selected_records

def save_output(selected_records, output_file):
    """Save the selected records to an output file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(list(selected_records.values()), file, indent=4)
    print(f"Selected records saved to {output_file}")
    print(f"Total unique calculation names found: {len(selected_records)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py input_file.json [output_file.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "selected_records.json"
    
    data = load_json_file(input_file)
    selected_records = select_records_by_calc_name(data)
    
    # Print summary
    print("Selected records by calculation name:")
    for calc_name in sorted(selected_records.keys()):
        print(f"- {calc_name}")
    
    save_output(selected_records, output_file)

if __name__ == "__main__":
    main()