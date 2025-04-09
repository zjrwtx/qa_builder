# Bioinformatics Code Generator

This script processes a JSON file containing bioinformatics problems, uses OpenAI's GPT-4o model to generate Python code solutions (prioritizing BioPython library), and saves the generated code to a new JSON file.

## Features

- Reads a JSON file containing bioinformatics problems
- Uses the CAMEL framework and GPT-4o model to generate Python code solutions
- Prioritizes BioPython library for solving problems
- Adds the generated code as a 'rationale' field to the JSON file
- Saves the updated JSON to a new file
- Utilizes multithreading to process multiple questions concurrently
- Supports batch processing for better resource utilization
- Automatically cleans the generated code to ensure only pure Python code is included
- Ensures all generated code stores final results in a 'result' variable and includes 'print(result)'

## Dependencies

- camel-ai>=0.1.0
- biopython>=1.81

## Installation

```bash
pip install -r requirements_code_generator.txt
```

## Usage

1. Ensure the `qa_data.json` file exists in the current directory with the correct format
2. Run the script:

```bash
python code_generator.py
```

3. The generated results will be saved to `qa_data_with_code.json`
4. Logs will be saved to `code_generator.log`

### Command Line Arguments

The script supports the following command line arguments:

```
--input INPUT       Input JSON file path (default: qa_data.json)
--output OUTPUT     Output JSON file path (default: qa_data_with_code.json)
--workers WORKERS   Number of worker threads (default: 10)
--batch_size SIZE   Number of questions to process in each batch (default: 1)
```

For example:

```bash
python code_generator.py --input my_data.json --output results.json --workers 5 --batch_size 2
```

## Code Cleaning

The script includes a code cleaning function that ensures only pure Python code is included in the output:

- Removes Markdown code block markers (```python, ```)
- Strips triple quotes (""") that might surround the code
- Removes explanatory text like "Python code:" prefixes
- Eliminates any non-code elements before or after the actual code
- Trims excess whitespace

This ensures that the rationale field contains only valid, executable Python code without any formatting artifacts.

## Result Variable Standardization

The script ensures all generated code follows a standardized output format:

- All solutions store their final answer in a variable named `result`
- All solutions include a `print(result)` statement at the end
- If the generated code doesn't follow this pattern, the script automatically adds or modifies the code to comply

This standardization makes the output more consistent and easier to process or execute. The script applies several heuristics to identify what should be the final result:

1. Looks for existing print statements to modify
2. Searches for return statements to convert
3. Examines variable assignments in the latter half of the code
4. Adds a placeholder if no suitable result can be determined

## Multithreading and Batch Processing

The script uses Python's ThreadPoolExecutor to process multiple questions concurrently, significantly improving processing speed when handling many questions.

- **Worker Threads**: Controls how many parallel threads are used. Default is 10, but will automatically adjust downward based on the number of questions.
- **Batch Size**: Controls how many questions are grouped together for processing by a single thread. Default is 1 (no batching). Increasing batch size can be helpful when working with a large number of small tasks to reduce overhead.

Optimal settings depend on your hardware and the OpenAI API rate limits:
- For CPU-bound processing: Set workers close to the number of CPU cores
- For API-limited processing: Adjust based on allowed requests per minute
- For memory-constrained environments: Lower the number of workers

## Input File Format

The input JSON file should be a list where each element is an object containing a `question` field, for example:

```json
[
  {
    "question": "Problem description...",
    "final_answer": "Problem answer..."
  },
  ...
]
```

## Output File Format

The output JSON file will add a `rationale` field to each object, with the value being the generated code:

```json
[
  {
    "question": "Problem description...",
    "final_answer": "Problem answer...",
    "rationale": "from Bio import SeqIO\n\ndef process_sequence(seq):\n    # Processing logic\n\nresult = process_sequence('AGTC')\nprint(result)"
  },
  ...
]
``` 