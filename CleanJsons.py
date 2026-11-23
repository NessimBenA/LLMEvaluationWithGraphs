import argparse
import json
import re
import sys
from pathlib import Path
from typing import Final, Literal, TypedDict

DEFAULT_REPLY_DIR: Final = "kg_json_outputs"
STANDARDIZED_PREFIX: Final = "standardized_"
MAX_JSON_DEPTH: Final = 200
FENCED_BLOCK: Final = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)
JSON_TOKEN: Final = re.compile(
    r"[ \t\n\r]*(?:"
    r'(?P<string>"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*")'
    r"|(?P<number>-?(?:0|[1-9][0-9]{0,4299})(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    r"|(?P<literal>true|false|null)"
    r"|(?P<punct>[{}\[\]:,]))"
)

ParserState = Literal["value", "value_or_close", "key", "key_or_close", "colon", "after", "done"]


class Node(TypedDict):
    id: str
    label: str


class Edge(TypedDict):
    source: str
    target: str
    label: str


class StandardGraph(TypedDict):
    nodes: list[Node]
    edges: list[Edge]


def closed(stack: list[str]) -> ParserState:
    return "after" if stack else "done"


def next_state(state: ParserState, kind: str, token: str, stack: list[str]) -> ParserState | None:
    if state in ("value", "value_or_close"):
        if state == "value_or_close" and token == "]":
            stack.pop()
            return closed(stack)
        if kind != "punct":
            return closed(stack)
        if token in ("{", "["):
            stack.append(token)
            return "key_or_close" if token == "{" else "value_or_close"
        return None
    if state in ("key", "key_or_close"):
        if state == "key_or_close" and token == "}":
            stack.pop()
            return closed(stack)
        return "colon" if kind == "string" else None
    if state == "colon":
        return "value" if token == ":" else None
    if token == ",":
        return "key" if stack[-1] == "{" else "value"
    if token == ("}" if stack[-1] == "{" else "]"):
        stack.pop()
        return closed(stack)
    return None


def is_valid_json(text: str) -> bool:
    stack: list[str] = []
    state: ParserState | None = "value"
    position = 0
    while state is not None and state != "done":
        match = JSON_TOKEN.match(text, position)
        if match is None or len(stack) > MAX_JSON_DEPTH:
            return False
        position = match.end()
        kind = match.lastgroup or ""
        state = next_state(state, kind, match.group(kind) if kind else "", stack)
    return state == "done" and text[position:].strip(" \t\n\r") == ""


def extract_json_text(content: str) -> str | None:
    candidates = [content, *(block.group(1) for block in FENCED_BLOCK.finditer(content))]
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if 0 <= start < end and is_valid_json(candidate[start : end + 1]):
            return candidate[start : end + 1]
    return None


def listed(value: object) -> list[object] | None:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    return None


def standardize_graph(data: object) -> tuple[StandardGraph | None, str]:
    if not isinstance(data, dict):
        return None, "the JSON is not an object"
    node_items = listed(data.get("nodes"))
    edge_items = listed(data.get("edges"))
    if node_items is None or edge_items is None:
        return None, "no 'nodes' or 'edges' list"
    nodes: list[Node] = []
    node_ids: set[str] = set()
    for item in node_items:
        if not isinstance(item, dict) or "id" not in item or "label" not in item:
            continue
        node_id = str(item["id"]).strip()
        if node_id and node_id not in node_ids:
            node_ids.add(node_id)
            nodes.append({"id": node_id, "label": str(item["label"]).strip()})
    if not nodes:
        return None, "no valid nodes"
    edges: list[Edge] = []
    pairs: set[tuple[str, str]] = set()
    for item in edge_items:
        if not isinstance(item, dict) or not all(key in item for key in ("source", "target", "label")):
            continue
        source = str(item["source"]).strip()
        target = str(item["target"]).strip()
        if source != target and source in node_ids and target in node_ids and (source, target) not in pairs:
            pairs.add((source, target))
            edges.append({"source": source, "target": target, "label": str(item["label"]).strip()})
    return {"nodes": nodes, "edges": edges}, ""


def standardize_reply(content: str) -> tuple[StandardGraph | None, str]:
    text = extract_json_text(content)
    if text is None:
        return None, "no valid JSON object found"
    return standardize_graph(json.loads(text))


def clean_directory(directory: Path, dry_run: bool) -> None:
    replies = sorted(path for path in directory.glob("*.json") if not path.name.startswith(STANDARDIZED_PREFIX))
    written = 0
    failed = 0
    for path in replies:
        target = path.with_name(STANDARDIZED_PREFIX + path.name)
        graph, reason = standardize_reply(path.read_text(encoding="utf-8", errors="replace"))
        if graph is None:
            failed += 1
            stale = target.exists()
            if stale and not dry_run:
                target.unlink()
            print(f"Failed  {path.name}: {reason}" + (f" (removed old {target.name})" if stale else ""))
            continue
        if not dry_run:
            target.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
        print(f"{'Would write' if dry_run else 'Wrote'}  {target.name}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    print(f"\n{len(replies)} replies, {written} standardized{' (dry run, nothing written)' if dry_run else ''}, {failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn each raw model reply into a standardized graph file, standardized_<name>.json, in the same folder."
    )
    parser.add_argument("directory", nargs="?", default=DEFAULT_REPLY_DIR, help=f"Folder of raw replies (default: {DEFAULT_REPLY_DIR}).")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written without writing anything.")
    args = parser.parse_args()
    directory = Path(str(args.directory))
    if not directory.is_dir():
        sys.exit(f"Folder not found: {directory}")
    clean_directory(directory, bool(args.dry_run))


if __name__ == "__main__":
    main()
