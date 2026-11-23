import argparse
import csv
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import requests
from dotenv import load_dotenv

from CleanJsons import DEFAULT_REPLY_DIR, StandardGraph, is_valid_json, standardize_reply

OPENROUTER_API_URL: Final = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LOG_FILE: Final = "kg_eval_log.csv"
DEFAULT_MAX_TOKENS: Final = 4 * 4096
DEFAULT_RETRIES: Final = 1
DEFAULT_TEMPERATURE: Final = 0.2
REQUEST_TIMEOUT_SECONDS: Final = 300
LOG_FIELDS: Final = (
    "timestamp",
    "subject",
    "model",
    "status",
    "num_retries_attempted",
    "num_retries_successful",
    "avg_num_nodes",
    "avg_num_edges",
    "avg_average_degree",
    "error_message",
    "last_api_response_preview",
)

KNOWLEDGE_GRAPH_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "description": "List of concepts, entities, or attributes in the knowledge graph.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique identifier for the node, used by edges to refer to it."},
                    "label": {"type": "string", "description": "Human-readable name of the concept, entity, or attribute."},
                },
                "required": ["id", "label"],
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array",
            "description": "List of directed relationships (edges) between nodes.",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "The 'id' of the source node."},
                    "target": {"type": "string", "description": "The 'id' of the target node."},
                    "label": {"type": "string", "description": "Description of the relationship from the source node to the target node."},
                },
                "required": ["source", "target", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes", "edges"],
    "additionalProperties": False,
}

JSON_INSTRUCTIONS: Final = """Please represent the generated graph strictly in JSON format adhering to the provided schema.

The JSON must be a single object with exactly two keys:
- "nodes": a list of objects, each with a unique string "id" and a human-readable string "label".
- "edges": a list of objects, each with a "source" and a "target", which are the "id" values of two different nodes listed in "nodes", and a string "label" describing the relationship from source to target.

Do not add any other keys. Output only the JSON object, with no text before or after it and no markdown code fences.

Schema:
{schema}"""


@dataclass(frozen=True)
class ModelReply:
    content: str
    finish_reason: str
    error: str


@dataclass(frozen=True)
class Settings:
    api_key: str
    max_tokens: int
    temperature: float
    attempts: int
    output_dir: Path


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")


def reply_path(output_dir: Path, subject: str, model: str, retry: int) -> Path:
    return output_dir / f"{sanitize_filename(subject)}_{sanitize_filename(model)}_retry_{retry}.json"


def build_prompt(concept: str) -> str:
    user_request = (
        f'Generate the most comprehensive knowledge graph possible for the concept "{concept}". '
        "Include as many relevant nodes (representing concepts, entities, attributes, etc.) "
        "and directed edges (representing relationships) as possible to capture the structure "
        "of knowledge about this concept."
    )
    instructions = JSON_INSTRUCTIONS.format(schema=json.dumps(KNOWLEDGE_GRAPH_SCHEMA, indent=2))
    return f'{user_request}\n\n{instructions}\n\nNow, generate the JSON knowledge graph for the concept: "{concept}"'


def read_reply(body: object) -> ModelReply:
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ModelReply("", "", f"no choices in the response: {json.dumps(body)[:300]}")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    finish_reason = str(choices[0].get("finish_reason"))
    if not isinstance(content, str) or not content.strip():
        return ModelReply("", finish_reason, f"empty reply (finish reason: {finish_reason})")
    return ModelReply(content.strip(), finish_reason, "")


def call_openrouter(prompt: str, model: str, api_key: str, max_tokens: int, temperature: float) -> ModelReply:
    print(f"      Calling OpenRouter (model: {model}, max tokens: {max_tokens}, schema enforced)...")
    response = requests.post(
        OPENROUTER_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Title": "KG-Eval-Tool"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "knowledge_graph_output", "strict": True, "schema": KNOWLEDGE_GRAPH_SCHEMA},
            },
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        return ModelReply("", "", f"HTTP {response.status_code}: {response.text[:300]}")
    if not is_valid_json(response.text):
        return ModelReply("", "", f"the response body is not JSON: {response.text[:300]}")
    return read_reply(json.loads(response.text))


def preview(text: str) -> str:
    return text if len(text) <= 200 else f"{text[:100]}...{text[-100:]}"


def log_results(logfile: Path, row: dict[str, str]) -> None:
    write_header = not logfile.is_file() or logfile.stat().st_size == 0
    with logfile.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summary_row(subject: str, model: str, attempts: int, graphs: list[StandardGraph], last_error: str, last_preview: str) -> dict[str, str]:
    node_counts = [float(len(graph["nodes"])) for graph in graphs]
    edge_counts = [float(len(graph["edges"])) for graph in graphs]
    degrees = [2 * edges / nodes for nodes, edges in zip(node_counts, edge_counts)]
    return {
        "timestamp": datetime.now().isoformat(),
        "subject": subject,
        "model": model,
        "status": "SUCCESS" if graphs else "ALL_RETRIES_FAILED",
        "num_retries_attempted": str(attempts),
        "num_retries_successful": str(len(graphs)),
        "avg_num_nodes": f"{statistics.mean(node_counts) if graphs else 0.0:.4f}",
        "avg_num_edges": f"{statistics.mean(edge_counts) if graphs else 0.0:.4f}",
        "avg_average_degree": f"{statistics.mean(degrees) if graphs else 0.0:.4f}",
        "error_message": "" if graphs else last_error,
        "last_api_response_preview": last_preview,
    }


def evaluate(subject: str, model: str, settings: Settings) -> dict[str, str]:
    graphs: list[StandardGraph] = []
    last_error = "No successful retries."
    last_preview = ""
    for retry in range(1, settings.attempts + 1):
        print(f"    Attempt {retry}/{settings.attempts}...")
        reply = call_openrouter(build_prompt(subject), model, settings.api_key, settings.max_tokens, settings.temperature)
        if reply.error:
            last_error = f"Retry {retry}: API error - {reply.error}"
            print(f"      {last_error}")
            continue
        path = reply_path(settings.output_dir, subject, model, retry)
        path.write_text(reply.content, encoding="utf-8")
        print(f"      Saved the reply to {path}")
        last_preview = preview(reply.content)
        graph, reason = standardize_reply(reply.content)
        if graph is None:
            last_error = f"Retry {retry}: parse error - {reason} (finish reason: {reply.finish_reason})"
            print(f"      {last_error}")
            continue
        print(f"      Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        graphs.append(graph)
    print(f"    Done: {len(graphs)}/{settings.attempts} attempts gave a graph.")
    return summary_row(subject, model, settings.attempts, graphs, last_error, last_preview)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask OpenRouter models for a JSON knowledge graph of each subject, save every reply and log the results.")
    parser.add_argument("--subjects", type=str, required=True, nargs="+", help="Subjects (concepts) to ask about.")
    parser.add_argument("--models", type=str, required=True, nargs="+", help="OpenRouter model ids.")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (default: OPENROUTER_API_KEY from the environment or .env).")
    parser.add_argument("--logfile", type=str, default=DEFAULT_LOG_FILE, help=f"CSV log, one row per subject and model (default: {DEFAULT_LOG_FILE}).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens per reply (default: {DEFAULT_MAX_TOKENS}).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).")
    parser.add_argument("--num-retries", type=int, default=DEFAULT_RETRIES, help=f"Attempts per subject and model; every attempt is run (default: {DEFAULT_RETRIES}).")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_REPLY_DIR, help=f"Folder for the raw replies (default: {DEFAULT_REPLY_DIR}).")
    args = parser.parse_args()
    if int(args.num_retries) < 1:
        sys.exit("--num-retries must be at least 1.")
    load_dotenv()
    api_key = str(args.api_key or os.getenv("OPENROUTER_API_KEY") or "")
    if not api_key:
        sys.exit("No OpenRouter API key: pass --api-key or set OPENROUTER_API_KEY in the environment or in .env.")
    settings = Settings(api_key, int(args.max_tokens), float(args.temperature), int(args.num_retries), Path(str(args.output_dir)))
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logfile = Path(str(args.logfile))
    subjects = [str(subject) for subject in args.subjects]
    models = [str(model) for model in args.models]
    print(f"Subjects: {', '.join(subjects)}\nModels: {', '.join(models)}\nReplies: {settings.output_dir}\nLog: {logfile}")
    total = len(subjects) * len(models)
    done = 0
    for subject in subjects:
        for model in models:
            done += 1
            print(f"\n[{done}/{total}] {model} on '{subject}'")
            log_results(logfile, evaluate(subject, model, settings))
    print(f"\nDone. Log: {logfile}. Replies: {settings.output_dir}")


if __name__ == "__main__":
    main()
