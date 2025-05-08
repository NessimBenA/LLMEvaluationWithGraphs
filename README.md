# LLM Knowledge Graph Generation Evaluation

Blog post on this benchmark here : https://nessimbena.github.io/blogposts/llm-benchmarking.html

A scientific toolkit for evaluating, analyzing, and comparing the knowledge representation capabilities of different Large Language Models (LLMs) through their ability to generate structured knowledge graphs.

## Research Goals

This project addresses several key research questions in LLM evaluation:

1. **Comparative Knowledge Representation**: How do different LLMs vary in their ability to represent domain knowledge in structured formats?
2. **Structural Comprehension**: How well do LLMs understand and represent complex relationships between concepts?
3. **Model Benchmarking**: Quantitative comparison of knowledge graph quality across different models (density, connectivity, accuracy)
4. **Domain Expertise Assessment**: Measuring domain-specific knowledge depth across multiple subjects

## Overview

This toolkit provides a rigorous methodological framework to:

1. Generate knowledge graphs from different LLM models via OpenRouter API using controlled prompts
2. Clean and standardize the JSON outputs for consistent evaluation
3. Create visualizations of the knowledge graphs for qualitative analysis
4. Benchmark and compare different models using graph-theoretical metrics
5. Analyze model performance across various domains of knowledge

## Components

- **GenerateJsons.py**: Generate knowledge graphs using various LLMs via OpenRouter
- **CleanJsons.py**: Clean and standardize JSON outputs from LLMs
- **GenerateGraphs.py**: Create visualizations of knowledge graphs
- **BenchmarkJsons.py**: Compare and benchmark different models' knowledge graph outputs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llmeval_cleaned.git
cd llmeval_cleaned

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Knowledge Graphs

```bash
python GenerateJsons.py --subjects "Machine Learning" --models "anthropic/claude-3-opus" "openai/gpt-4o-latest" --api-key YOUR_API_KEY --num-retries 3
```

Required arguments:
- `--subjects`: List of concepts/subjects to generate knowledge graphs for
- `--models`: List of OpenRouter model identifiers
- `--api-key`: Your OpenRouter API key (or set via environment variable)

Optional arguments:
- `--logfile`: Path to output CSV log (default: kg_eval_log.csv)
- `--max-tokens`: Maximum tokens for LLM response (default: 16384)
- `--temperature`: Sampling temperature (default: 0.2)
- `--num-retries`: Number of generation attempts per subject/model (default: 1)
- `--output-dir`: Directory to save JSON files (default: kg_json_outputs)

### 2. Clean JSON Files

```bash
python CleanJsons.py kg_json_outputs --dry-run
# Remove --dry-run to actually modify files
```

### 3. Generate Graph Visualizations

```bash
python GenerateGraphs.py kg_json_outputs graph_images -fmt svg
```

### 4. Benchmark Models

```bash
python BenchmarkJsons.py
```

The script reads the log CSV file and generates benchmark visualizations.

## Environment Variables

Store your API key and other configuration in a `.env` file:

```
OPENROUTER_API_KEY=your_api_key_here
```

## License

[MIT License](LICENSE)
