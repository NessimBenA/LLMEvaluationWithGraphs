import requests
import json
import networkx as nx
import os
import argparse
import sys
import csv
from datetime import datetime
import time # For potential rate limiting
import re   # For filename sanitization
import statistics # For averaging

# --- Configuration ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LOG_FILE = "kg_eval_log.csv"
DEFAULT_MAX_TOKENS = 4*4096
DEFAULT_RETRIES = 1 # Default to 1 try (no retries)
DEFAULT_OUTPUT_DIR = "kg_json_outputs"

# --- Define the expected JSON Schema (No changes needed here) ---
KNOWLEDGE_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "description": "List of concepts, entities, or attributes in the knowledge graph.",
            "items": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Unique identifier for the node..." },
                    "label": { "type": "string", "description": "Human-readable name..." }
                },
                "required": ["id", "label"],
                "additionalProperties": False # For node items (already added)
            }
        },
        "edges": {
            "type": "array",
            "description": "List of directed relationships (edges) between nodes.",
            "items": {
                "type": "object",
                "properties": {
                    "source": { "type": "string", "description": "The 'id' of the source node." },
                    "target": { "type": "string", "description": "The 'id' of the target node." },
                    "label": { "type": "string", "description": "Description of the relationship..." }
                },
                "required": ["source", "target", "label"],
                "additionalProperties": False # For edge items (already added)
            }
        }
    },
    "required": ["nodes", "edges"],
    "additionalProperties": False  # <<< ADDED HERE for the top-level object
}

# --- Helper Functions ---

def get_api_key(args_api_key):
    # (No changes needed here)
    api_key = args_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OpenRouter API key not provided via --api-key or OPENROUTER_API_KEY env var.", file=sys.stderr)
        sys.exit(1)
    return api_key

def sanitize_filename(name: str) -> str:
    """Removes or replaces characters unsafe for filenames."""
    # Remove characters that are definitely problematic
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Replace slashes often used in model names
    name = name.replace("/", "-")
    # Limit length if necessary (optional)
    # max_len = 100
    # if len(name) > max_len:
    #     name = name[:max_len]
    return name

def generate_knowledge_graph_prompt_v2(concept: str) -> str:
    # (No changes needed here)
    user_request = (
        f'Generate the most comprehensive knowledge graph possible for the concept "{concept}". '
        'Include as many relevant nodes (representing concepts, entities, attributes, etc.) '
        'and directed edges (representing relationships) as possible to capture the structure '
        f'of knowledge about this concept.'
    )
    json_instructions = """
Please represent the generated graph strictly in JSON format adhering to the provided schema...
... (rest of instructions) ...
"""
    prompt = f"""
{user_request}

{json_instructions}

Now, generate the JSON knowledge graph for the concept: "{concept}"
"""
    return prompt.strip()


def call_openrouter(prompt: str, model: str, api_key: str, max_tokens: int, temperature: float) -> tuple[str | None, str | None]:
    # (No changes needed in the core API call logic itself)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "YOUR_SITE_URL",
        "X-Title": "KG-Eval-Tool",
    }
    response_format = {
        "type": "json_schema",
        "json_schema": { "name": "knowledge_graph_output", "strict": True, "schema": KNOWLEDGE_GRAPH_SCHEMA }
    }
    data = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "response_format": response_format, "max_tokens": max_tokens, "temperature": temperature,
    }

    print(f"      Calling OpenRouter API (Model: {model}, Max Tokens: {max_tokens}, Schema Enforced)...")
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content")
        finish_reason = response_json.get("choices", [{}])[0].get("finish_reason")

        if not content and finish_reason != "stop":
            if finish_reason == "length":
                 error_msg = f"API response finished due to 'length' (max_tokens: {max_tokens} likely reached)."
            else:
                 error_msg = f"API response missing content. Finish reason: '{finish_reason}'. Response: {str(response_json)[:200]}..."
            print(f"      Error: {error_msg}", file=sys.stderr)
            return None, error_msg
        if not content:
            error_msg = f"API response missing content despite finish_reason='stop'. Response: {str(response_json)[:200]}..."
            print(f"      Warning: {error_msg}", file=sys.stderr)

        print(f"      API Call Successful (Model: {model}, Finish Reason: {finish_reason}).")
        return content.strip() if content else None, None

    # --- Error handling blocks remain the same ---
    except requests.exceptions.Timeout: error_msg = "API request timed out."; print(f"      Error: {error_msg}", file=sys.stderr); return None, error_msg
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {e.response.status_code} {e.response.reason}"
        try: error_details = e.response.json(); error_msg += f" - Details: {error_details}"
        except json.JSONDecodeError: error_msg += f" - Body: {e.response.text[:200]}..."
        print(f"      Error: {error_msg}", file=sys.stderr)
        if e.response.status_code == 400 and ("response_format" in e.response.text or "json_schema" in e.response.text): print(f"      Hint: Model '{model}' might not support 'json_schema' structured outputs.", file=sys.stderr)
        return None, error_msg
    except requests.exceptions.RequestException as e: error_msg = f"Request Exception: {e}"; print(f"      Error: {error_msg}", file=sys.stderr); return None, error_msg
    except (IndexError, KeyError) as e: response_data_str = str(response_json) if 'response_json' in locals() else "N/A"; error_msg = f"Unexpected API response structure: {e}. Response: {response_data_str[:500]}..."; print(f"      Error: {error_msg}", file=sys.stderr); return None, error_msg
    except Exception as e: error_msg = f"Unexpected error during API call: {type(e).__name__} - {e}"; print(f"      Error: {error_msg}", file=sys.stderr); return None, error_msg


def parse_llm_response_to_graph(response_content: str) -> tuple[nx.DiGraph | None, str | None]:
    """
    Parses the LLM's JSON response. Handles potential markdown fences,
    preceding text, and cases where 'nodes'/'edges' are objects instead of lists.
    Returns: (graph, error_message)
    """
    # --- Initial part (cleaning, brace extraction) remains the same ---
    print("        Parsing LLM response...")
    if not response_content: return None, "Empty response content received for parsing."
    original_content = response_content
    processed_content = response_content.strip()
    cleaning_step = "original"
    if processed_content.startswith("```json") and processed_content.endswith("```"):
        processed_content = processed_content[7:-3].strip(); cleaning_step = "cleaned ```json fences"; print(f"          {cleaning_step}.")
    elif processed_content.startswith("```") and processed_content.endswith("```"):
        processed_content = processed_content[3:-3].strip(); cleaning_step = "cleaned ``` fences"; print(f"          {cleaning_step}.")
    if not processed_content: return None, f"Response contained only markdown fences or whitespace ({cleaning_step})."
    first_brace = processed_content.find('{'); last_brace = processed_content.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        potential_json = processed_content[first_brace : last_brace + 1]
        cleaning_step = f"{cleaning_step}, then extracted bracket content" if cleaning_step != "original" else "extracted bracket content"
        print(f"          {cleaning_step}.")
    else:
        potential_json = processed_content
        print(f"          Warning: Could not find enclosing braces after {cleaning_step}.", file=sys.stderr)
    # --- End Initial part ---

    try:
        if not potential_json.strip(): return None, f"Extracted content between braces was empty or whitespace ({cleaning_step})."
        data = json.loads(potential_json)

        # --- Structure Validation (with tolerance) ---
        if not isinstance(data, dict): return None, f"Parsed data is not a dictionary (type: {type(data)})."
        if "nodes" not in data or "edges" not in data: return None, "Parsed JSON missing required 'nodes' or 'edges' key."

        # --- Get Node Items (Tolerate List or Dict) ---
        node_items = []
        if isinstance(data["nodes"], list):
            node_items = data["nodes"]
            print("          'nodes' field is a list (as expected).")
        elif isinstance(data["nodes"], dict):
            # If it's a dict, iterate through its values
            node_items = list(data["nodes"].values())
            print("          Warning: 'nodes' field is an object/dict, processing its values.", file=sys.stderr)
        else:
            return None, f"'nodes' key is not a list or object/dict (type: {type(data['nodes'])})."

        # --- Get Edge Items (Tolerate List or Dict) ---
        edge_items = []
        if isinstance(data["edges"], list):
            edge_items = data["edges"]
            print("          'edges' field is a list (as expected).")
        elif isinstance(data["edges"], dict):
            # If it's a dict, iterate through its values
            edge_items = list(data["edges"].values())
            print("          Warning: 'edges' field is an object/dict, processing its values.", file=sys.stderr)
        else:
             return None, f"'edges' key is not a list or object/dict (type: {type(data['edges'])})."


        # --- Graph Construction (using node_items and edge_items) ---
        G = nx.DiGraph()
        node_ids = set()
        nodes_added, edges_added = 0, 0

        # Add nodes from node_items
        for i, node_data in enumerate(node_items): # Iterate through the extracted list/values
            if not isinstance(node_data, dict) or "id" not in node_data or "label" not in node_data: print(f"          Warning: Skipping invalid node data at index {i}", file=sys.stderr); continue
            node_id = str(node_data["id"]).strip(); node_label = str(node_data["label"]).strip()
            if not node_id: print(f"          Warning: Skipping node with empty ID at index {i}", file=sys.stderr); continue
            if node_id in node_ids: print(f"          Warning: Duplicate node ID '{node_id}'.", file=sys.stderr); continue
            G.add_node(node_id, label=node_label); node_ids.add(node_id); nodes_added += 1

        # Add edges from edge_items
        for i, edge_data in enumerate(edge_items): # Iterate through the extracted list/values
            if not isinstance(edge_data, dict) or "source" not in edge_data or "target" not in edge_data or "label" not in edge_data: print(f"          Warning: Skipping invalid edge data at index {i}", file=sys.stderr); continue
            source_id = str(edge_data["source"]).strip(); target_id = str(edge_data["target"]).strip(); edge_label = str(edge_data["label"]).strip()
            if not source_id or not target_id: print(f"          Warning: Skipping edge with empty source/target ID at index {i}", file=sys.stderr); continue
            if source_id not in node_ids: print(f"          Warning: Edge source node '{source_id}' not found.", file=sys.stderr); continue
            if target_id not in node_ids: print(f"          Warning: Edge target node '{target_id}' not found.", file=sys.stderr); continue
            if source_id == target_id: print(f"          Warning: Skipping self-loop edge for node '{source_id}'.", file=sys.stderr); continue
            G.add_edge(source_id, target_id, label=edge_label); edges_added += 1

        print(f"        Graph Constructed (Nodes Added: {nodes_added}, Edges Added: {edges_added})")
        if G.number_of_nodes() == 0 and (len(node_items) > 0 or len(edge_items) > 0): return None, "Graph construction resulted in an empty graph."
        return G, None # Success

    # --- Error handling remains the same ---
    except json.JSONDecodeError as e:
        error_msg = f"JSON Decode Error after {cleaning_step}: {e}. Preview: {potential_json[:200]}..."
        print(f"        Error: {error_msg}", file=sys.stderr)
        if cleaning_step != "original" and potential_json != original_content: print(f"        Original Response Preview: {original_content[:200]}...", file=sys.stderr)
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during graph parsing after {cleaning_step}: {type(e).__name__} - {e}"
        print(f"        Error: {error_msg}", file=sys.stderr)
        return None, error_msg


# --- Rest of the script remains the same ---
# (Make sure to replace the old schema and parser function)
def calculate_graph_metrics(G: nx.DiGraph) -> dict:
    # (No changes needed here)
    print("          Calculating graph metrics...")
    metrics = {}
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    metrics["num_nodes"] = num_nodes
    metrics["num_edges"] = num_edges
    if num_nodes > 0:
        degrees = [d for n, d in G.degree()]
        metrics["average_degree"] = sum(degrees) / num_nodes
    else:
        metrics["average_degree"] = 0.0
    print("          Metrics Calculated.")
    return metrics

def log_results(logfile: str, data: dict):
    """Appends a dictionary of results to a CSV log file."""
    file_exists = os.path.isfile(logfile)
    # --- Added retry counts to fieldnames ---
    fieldnames = [
        'timestamp', 'subject', 'model', 'status',
        'num_retries_attempted', 'num_retries_successful', # Added
        'avg_num_nodes', 'avg_num_edges', 'avg_average_degree', # Changed names for clarity
        'error_message', 'last_api_response_preview' # Changed name
        ]

    try:
        with open(logfile, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')

            if not file_exists or os.path.getsize(logfile) == 0:
                writer.writeheader()

            log_entry = {field: data.get(field, '') for field in fieldnames}
            log_entry['timestamp'] = datetime.now().isoformat()

            # Format floats for consistency
            for key in ['avg_num_nodes', 'avg_num_edges', 'avg_average_degree']:
                 if isinstance(log_entry.get(key), float):
                     log_entry[key] = f"{log_entry[key]:.4f}"

            writer.writerow(log_entry)

    except IOError as e:
        print(f"Error writing to log file {logfile}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error during logging: {e}", file=sys.stderr)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate LLM knowledge graph generation with retries and JSON saving.")
    parser.add_argument("--subjects", type=str, required=True, nargs='+', help="List of subjects/concepts.")
    parser.add_argument("--models", type=str, required=True, nargs='+', help="List of OpenRouter model identifiers.")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (overrides env var).")
    parser.add_argument("--logfile", type=str, default=DEFAULT_LOG_FILE, help=f"Path to the CSV log file (default: {DEFAULT_LOG_FILE}).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens for LLM response (default: {DEFAULT_MAX_TOKENS}).")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for LLM.")
    # --- New arguments ---
    parser.add_argument("--num-retries", type=int, default=DEFAULT_RETRIES, help=f"Number of generation attempts per subject/model (default: {DEFAULT_RETRIES}).")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save generated JSON files (default: {DEFAULT_OUTPUT_DIR}).")
    # parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls.")

    args = parser.parse_args()

    # Validate retries argument
    if args.num_retries < 1:
        print("Error: --num-retries must be at least 1.", file=sys.stderr)
        sys.exit(1)

    api_key = get_api_key(args.api_key)

    # --- Create output directory if it doesn't exist ---
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory for JSON files: {args.output_dir}")
    except OSError as e:
        print(f"Error creating output directory '{args.output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Starting Batch Knowledge Graph Evaluation ---")
    print(f"Subjects: {', '.join(args.subjects)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Logging results to: {args.logfile}")
    print(f"Retries per combination: {args.num_retries}")
    print(f"Default Max Tokens: {args.max_tokens}")
    print(f"---")

    total_combinations = len(args.subjects) * len(args.models)
    current_combination = 0

    for subject in args.subjects:
        sanitized_subject = sanitize_filename(subject)
        print(f"\nProcessing Subject: '{subject}'")

        for model in args.models:
            current_combination += 1
            sanitized_model = sanitize_filename(model)
            print(f"  [{current_combination}/{total_combinations}] Testing Model: '{model}' for Subject: '{subject}' ({args.num_retries} retries)")

            successful_metrics_list = []
            last_error_message = "No successful retries."
            last_api_response_preview = ""

            # --- Retry Loop ---
            for retry_num in range(1, args.num_retries + 1):
                print(f"    Attempt {retry_num}/{args.num_retries}...")
                attempt_success = False

                # 1. Generate Prompt
                prompt = generate_knowledge_graph_prompt_v2(subject)

                # 2. Call LLM
                llm_response, api_error = call_openrouter(prompt, model, api_key, args.max_tokens, args.temperature)

                if api_error or llm_response is None:
                    last_error_message = f"Retry {retry_num}: API Error - {api_error or 'Empty content'}"
                    print(f"      {last_error_message}")
                    # Optional delay even on failure if hitting rate limits
                    # if args.delay > 0: time.sleep(args.delay)
                    continue # Go to next retry

                # --- Save successful JSON response ---
                json_filename = f"{sanitized_subject}_{sanitized_model}_retry_{retry_num}.json"
                json_filepath = os.path.join(args.output_dir, json_filename)
                try:
                    with open(json_filepath, 'w', encoding='utf-8') as f_json:
                        f_json.write(llm_response)
                    print(f"        Saved JSON to {json_filepath}")
                except IOError as e:
                    print(f"        Warning: Failed to save JSON to {json_filepath}: {e}", file=sys.stderr)
                    # Continue processing even if saving failed, but log warning

                # Store preview for logging in case this is the last attempt
                last_api_response_preview = llm_response[:100] + "..." + llm_response[-100:] if len(llm_response) > 200 else llm_response

                # 3. Parse Response and Build Graph
                graph, parse_error = parse_llm_response_to_graph(llm_response)

                if parse_error or not graph:
                    last_error_message = f"Retry {retry_num}: Parse Error - {parse_error or 'Empty graph'}"
                    print(f"      {last_error_message}")
                    continue # Go to next retry

                # 4. Calculate Metrics
                try:
                    metrics = calculate_graph_metrics(graph)
                    successful_metrics_list.append(metrics) # Store metrics from successful retry
                    attempt_success = True
                    last_error_message = "" # Clear last error on success
                except Exception as e:
                     last_error_message = f"Retry {retry_num}: Metric Error - {e}"
                     print(f"      {last_error_message}", file=sys.stderr)
                     continue # Go to next retry

                # Optional delay between successful calls
                # if args.delay > 0 and retry_num < args.num_retries:
                #     print(f"    Waiting {args.delay}s...")
                #     time.sleep(args.delay)
            # --- End Retry Loop ---

            # --- Aggregate Results ---
            num_successful = len(successful_metrics_list)
            final_log_data = {
                'subject': subject,
                'model': model,
                'num_retries_attempted': args.num_retries,
                'num_retries_successful': num_successful,
                'error_message': last_error_message, # Log last error if all failed
                'last_api_response_preview': last_api_response_preview # Log preview from last attempt
            }

            if num_successful > 0:
                # Calculate average metrics
                avg_nodes = statistics.mean([m['num_nodes'] for m in successful_metrics_list])
                avg_edges = statistics.mean([m['num_edges'] for m in successful_metrics_list])
                avg_degree = statistics.mean([m['average_degree'] for m in successful_metrics_list])

                final_log_data['status'] = 'SUCCESS'
                final_log_data['avg_num_nodes'] = avg_nodes
                final_log_data['avg_num_edges'] = avg_edges
                final_log_data['avg_average_degree'] = avg_degree
                final_log_data['error_message'] = '' # Clear error message if at least one succeeded

                print(f"    Finished Retries. Success Rate: {num_successful}/{args.num_retries}. Avg Degree: {avg_degree:.4f}")

            else:
                # All retries failed
                final_log_data['status'] = 'ALL_RETRIES_FAILED'
                final_log_data['avg_num_nodes'] = 0.0
                final_log_data['avg_num_edges'] = 0.0
                final_log_data['avg_average_degree'] = 0.0
                # Keep last_error_message and last_api_response_preview

                print(f"    Finished Retries. All {args.num_retries} attempts failed.")


            # 5. Log Aggregated Results
            log_results(args.logfile, final_log_data)

    print("\n--- Batch Evaluation Complete ---")
    print(f"Results logged to: {args.logfile}")
    print(f"Generated JSON files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()