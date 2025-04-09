import json
import os
import logging
import threading
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("code_generator.log"),
        logging.StreamHandler()
    ]
)

def clean_code(code_text):
    """Clean the code text to remove non-code elements
    
    This function:
    1. Removes triple quotes and code block markers
    2. Removes markdown code block markers (```python and ```)
    3. Removes other non-code elements
    4. Trims leading/trailing whitespace
    """
    # Remove code block markers if present
    code_text = re.sub(r'```python|```py|```|`', '', code_text)
    
    # Remove triple quotes that might be around the code
    code_text = re.sub(r'^"""|"""$', '', code_text)
    
    # Remove "Python code:" or similar prefixes
    code_text = re.sub(r'^.*?[Pp]ython [Cc]ode:?\s*', '', code_text)
    
    # Remove explanations before or after the code
    code_text = re.sub(r'^.*?Here\'s the code:?\s*', '', code_text)
    code_text = re.sub(r'\n\s*This code will.*?$', '', code_text)
    
    # Trim leading/trailing whitespace
    code_text = code_text.strip()
    
    return code_text

def process_question(item, agent, index, total):
    """Process a single question using the LLM agent"""
    question = item['question']
    logging.info(f"Processing question {index+1}/{total}")
    
    try:
        # Construct prompt requesting BioPython-based solutions
        prompt = f"""
        Based on the following bioinformatics problem, write Python code to solve it.
        Use BioPython library whenever possible.
        Return only the pure Python code without any explanations, markdown formatting, or code block markers.
        Do not include triple quotes, comments explaining what the code does, or any non-code elements.
        
        IMPORTANT: Your code MUST store the final answer in a variable named 'result' and include 'print(result)' at the end.
        
        Problem: {question}
        """
        
        # Get response
        response = agent.step(prompt)
        
        # Extract code, clean it and add to data
        code = response.msgs[0].content
        clean_code_result = clean_code(code)
        item['rationale'] = clean_code_result
        logging.info(f"Question {index+1} processed successfully")
        return item
    except Exception as e:
        logging.error(f"Error processing question {index+1}: {e}")
        item['rationale'] = "Processing error"
        return item

def process_batch(batch_items, agent, start_index, total):
    """Process a batch of questions"""
    results = []
    for i, item in enumerate(batch_items):
        idx = start_index + i
        results.append(process_question(item, agent, idx, total))
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate code for bioinformatics problems using LLM')
    parser.add_argument('--input', type=str, default='qa_data.json', help='Input JSON file path')
    parser.add_argument('--output', type=str, default='qa_data_with_code.json', help='Output JSON file path')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of questions to process in each batch')
    args = parser.parse_args()
    
    try:
        # Read JSON file
        input_file = args.input
        output_file = args.output
        
        logging.info(f"Starting to process file: {input_file}")
        
        if not os.path.exists(input_file):
            logging.error(f"Input file {input_file} does not exist")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:  # Added explicit UTF-8 encoding
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {e}")
                return
        
        # Initialize model
        logging.info("Initializing LLM model")
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig().as_dict(),
        )
        
        # Initialize agent
        camel_agent = ChatAgent(model=model)
        
        # Process questions using thread pool
        total_questions = len(data)
        
        # Determine optimal number of workers
        max_workers = min(args.workers, total_questions)
        batch_size = args.batch_size
        
        logging.info(f"Processing {total_questions} questions using {max_workers} threads with batch size {batch_size}")
        
        # Prepare batches
        batches = []
        for i in range(0, total_questions, batch_size):
            batch = data[i:i+batch_size]
            batches.append((batch, i))
        
        processed_data = []
        
        # 创建进度条
        pbar = tqdm(total=total_questions, desc="处理问题进度")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches for processing
            futures = []
            for batch_items, start_idx in batches:
                if batch_size == 1:
                    # Process single item directly
                    futures.append(
                        executor.submit(process_question, batch_items[0], camel_agent, start_idx, total_questions)
                    )
                else:
                    # Process batch
                    futures.append(
                        executor.submit(process_batch, batch_items, camel_agent, start_idx, total_questions)
                    )
            
            # Collect results
            for future in futures:
                result = future.result()
                if batch_size == 1:
                    processed_data.append(result)
                    pbar.update(1)  # 更新进度条
                else:
                    processed_data.extend(result)
                    pbar.update(len(result))  # 更新进度条
        
        # 关闭进度条
        pbar.close()
        
        # Save updated JSON to new file
        try:
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            logging.info(f"Results saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving file: {e}")
    
    except Exception as e:
        logging.error(f"Error during execution: {e}")

if __name__ == "__main__":
    main()