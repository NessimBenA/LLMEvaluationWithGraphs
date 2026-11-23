# LLM Knowledge Graph Generation Evaluation

Blog post on this benchmark: [Benchmarking LLMs Through Knowledge Graph Analysis](https://nessimbena.github.io/blogposts/llm-benchmarking.html)

The idea is simple: ask a model for the most comprehensive knowledge graph it can produce on a subject, then look at how connected that graph is. If a model understands a subject, it should be able to link its concepts to each other. There is no fixed dataset, no LLM judge and no human in the loop. The code does not check whether the facts in a graph are true.

The pipeline has four steps, and each one reads what the previous one writes:

1. `GenerateJsons.py` asks each model, through OpenRouter, for a JSON knowledge graph of each subject, saves every reply and logs the results.
2. `CleanJsons.py` turns each raw reply into a standardized graph file.
3. `GenerateGraphs.py` draws each standardized graph with Graphviz.
4. `BenchmarkJsons.py` scores each graph on six graph metrics and ranks the models.

Only step 1 calls an API, and it calls paid models.

## Setup

```bash
git clone https://github.com/NessimBenA/LLMEvaluationWithGraphs.git
cd LLMEvaluationWithGraphs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Step 3 also needs the Graphviz `dot` program: `sudo apt install graphviz` on Debian and Ubuntu, `brew install graphviz` on macOS.

Step 1 needs an OpenRouter API key. Copy `.env.example` to `.env` and put your key after `OPENROUTER_API_KEY=`, or export `OPENROUTER_API_KEY` in your shell, or pass `--api-key`.

## Run

### 1. Generate the graphs

```bash
python GenerateJsons.py --subjects "Physical Sciences" "Life Sciences" --models openai/gpt-4.1 anthropic/claude-3.7-sonnet --num-retries 5
```

The run behind the blog post used five subjects and seven models, with five attempts each:

```bash
python GenerateJsons.py \
  --subjects "Physical Sciences" "Life Sciences" "Mathematics & Computer Science" "Engineering & Applied Sciences" "Social & Behavioral Sciences" \
  --models google/gemini-2.5-pro-preview-03-25 google/gemini-2.5-flash-preview openai/o4-mini-high openai/gpt-4.1 openai/gpt-4.1-nano deepseek/deepseek-chat-v3-0324 anthropic/claude-3.7-sonnet \
  --num-retries 5
```

Model ids change over time, so check them on [openrouter.ai/models](https://openrouter.ai/models) before a run.

`--num-retries` is the number of attempts per subject and model. Every attempt is run, so `--num-retries 5` gives up to five graphs per subject and model. An attempt succeeds when its reply gives a graph under the cleaning rules of step 2.

It writes:

- `kg_json_outputs/<subject>_<model>_retry_<n>.json`: the raw reply of each attempt. Characters that are unsafe in file names are removed and spaces become underscores, so `openai/gpt-4.1` becomes `openaigpt-4.1`.
- `kg_eval_log.csv`: one row per subject and model with the status, the number of attempts and of successful ones, the average node count, edge count and node degree over the successful attempts, the last error and a preview of the last reply. New runs are appended.

Other options: `--max-tokens` (default 16384), `--temperature` (default 0.2), `--output-dir` (default `kg_json_outputs`), `--logfile` (default `kg_eval_log.csv`).

### 2. Clean the replies

```bash
python CleanJsons.py kg_json_outputs
```

For each raw reply `<name>.json`, it writes `standardized_<name>.json` in the same folder and leaves the raw reply as it is. Add `--dry-run` to see what it would write.

The cleaning:

- takes the JSON out of the surrounding text and markdown fences (from the first `{` to the last `}`, or the content of a fenced block);
- accepts `nodes` and `edges` given as lists or as objects;
- keeps the nodes that have an `id` and a `label`, and the first node for each `id`;
- keeps the edges that have a `source`, a `target` and a `label`, that join two different nodes of the graph, and that do not repeat an earlier edge between the same two nodes;
- keeps only `id` and `label` on nodes, and `source`, `target` and `label` on edges.

A reply fails when it holds no valid JSON object (for example when it was cut off at the token limit) or when no valid node is left. If a failed reply had a standardized file from an earlier run, that file is removed.

### 3. Draw the graphs

```bash
python GenerateGraphs.py kg_json_outputs graph_images -fmt svg
```

It writes one image per standardized file to `graph_images/`, named after the reply: `<subject>_<model>_retry_<n>.svg`. Formats: `svg` (default), `png`, `pdf`, `jpg`, `gif`. Images that already exist are skipped unless you pass `-f`. Add `--skip_errors` to keep going when a graph fails to render.

### 4. Rank the models

```bash
python BenchmarkJsons.py
```

It reads `kg_eval_log.csv` and the standardized files in `kg_json_outputs/` (change them with `--log` and `--json-dir`). For each subject and model, it uses the attempts of the latest run in the log. It writes to `benchmark_results/` (change it with `--output-dir`):

- `graph_metrics.csv`: the six metrics of every graph;
- `subject_scores.csv`: the metrics averaged over the attempts of each subject and model, and the composite score;
- `model_ranking.csv`: the mean composite score of each model, and on how many subjects it was scored;
- `model_ranking.png` and `subject_scores.png`: the same numbers as charts.

It also prints the ranking.

## Scoring

The metrics and weights are the ones in the blog post:

| Metric | What it measures | Weight |
|---|---|---|
| Average node degree | Connections per node: 2 × edges / nodes | 2.0 |
| Node count | Concepts and entities in the graph | 1.0 |
| Edge count | Relationships in the graph | 1.0 |
| Graph density | Edges / (nodes × (nodes − 1)), the share of possible directed edges that are present | 0.8 |
| Connected components | Number of weakly connected parts (lower is better) | 1.0 |
| Largest component ratio | Nodes in the largest component / all nodes | 0.8 |

For each subject:

1. Each metric is averaged over the successful attempts of each model.
2. Each metric is scaled to [0, 1] across the models: (value − min) / (max − min). For connected components the scale is flipped, so 1 is always the best. When all models have the same value, they all get 1.
3. The composite score is the weighted average of the six scaled metrics (the weights add up to 6.6).

The models are ranked by their mean composite score over the subjects. A subject on which a model produced no graph is left out of its mean, and `model_ranking.csv` shows on how many subjects each model was scored.

## The prompt

Each model gets the prompt quoted in the blog post:

> Generate the most comprehensive knowledge graph possible for the concept "{concept}". Include as many relevant nodes (representing concepts, entities, attributes, etc.) and directed edges (representing relationships) as possible to capture the structure of knowledge about this concept.

followed by instructions on the JSON format, the JSON schema itself, and a last line asking for the graph. The schema is also sent as the `response_format` of the request, in OpenRouter's strict JSON schema mode.

Note: before this version, the prompt sent the first sentence of the JSON instructions and then the literal line `... (rest of instructions) ...`. The original wording of those instructions could not be recovered. The instructions after that first sentence, and the descriptions of the schema fields, were rewritten to be complete, so graphs generated now can differ from the ones made with the earlier prompt.

## Type checks

```bash
pip install mypy types-requests types-networkx
mypy --strict *.py
```

## License

MIT, see [LICENSE](LICENSE).
