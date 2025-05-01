import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import os
import re # For filename sanitization (reuse from main script)
import sys
import matplotlib.colors as mcolors
import math
# --- Configuration ---
CSV_FILE = "broad_eval_results.csv"
JSON_DIR = "kg_outputs_broad_eval"
NUM_TOP_GRAPHS = 3

# --- Helper Function (reuse from main script) ---
def sanitize_filename(name: str) -> str:
    """Removes or replaces characters unsafe for filenames."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    name = name.replace("/", "-")
    name = name.replace("&", "and") # Handle '&' specifically
    return name

# --- Function to Load and Prepare Data ---
def load_and_prepare_data(csv_path):
    """Loads CSV, converts types, and filters for successful runs."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert metric columns to numeric, coercing errors to NaN
    metric_cols = ['avg_num_nodes', 'avg_num_edges', 'avg_average_degree']
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert retry counts to integers
    retry_cols = ['num_retries_attempted', 'num_retries_successful']
    for col in retry_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)


    # Filter for successful runs for ranking/plotting metrics
    df_success = df[df['status'] == 'SUCCESS'].copy()
    df_success.dropna(subset=metric_cols, inplace=True) # Drop rows where metrics couldn't be parsed

    print(f"Loaded {len(df)} records, {len(df_success)} successful runs.")
    return df, df_success # Return both original and success-filtered

# --- Plotting Functions ---

def plot_ranking_per_concept(df_success):
    """Plots average degree for each model, grouped by concept."""
    if df_success.empty:
        print("No successful runs to plot ranking per concept.")
        return

    plt.figure(figsize=(15, 8))
    sns.barplot(data=df_success, x='subject', y='avg_average_degree', hue='model', palette='cubehelix')

    plt.title('Model Ranking by Average Node Degree per Concept')
    plt.xlabel('Concept (Subject)')
    plt.ylabel('Average Node Degree (Avg. over Retries)')
    plt.xticks(rotation=15, ha='right') # Rotate labels slightly
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig('trial1.jpg')

def plot_overall_ranking(df_success):
    """Plots overall average degree per model across all concepts."""
    if df_success.empty:
        print("No successful runs to plot overall ranking.")
        return

    # Calculate overall average degree per model
    overall_avg = df_success.groupby('model')['avg_average_degree'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=overall_avg.index, y=overall_avg.values, palette='magma')

    plt.title('Overall Model Ranking by Average Node Degree (Across All Concepts)')
    plt.xlabel('Model')
    plt.ylabel('Overall Average Node Degree')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('trial2.jpg')

# --- Graph Loading and Plotting Functions ---

def create_graph_from_json(json_data: dict) -> nx.DiGraph | None:
    """Creates a NetworkX graph from the parsed JSON data (list/dict tolerant)."""
    # --- Get Node Items (Tolerate List or Dict) ---
    node_items = []
    if isinstance(json_data.get("nodes"), list):
        node_items = json_data["nodes"]
    elif isinstance(json_data.get("nodes"), dict):
        node_items = list(json_data["nodes"].values())
    else:
        print("Error: 'nodes' key is not a list or object/dict in JSON.", file=sys.stderr)
        return None

    # --- Get Edge Items (Tolerate List or Dict) ---
    edge_items = []
    if isinstance(json_data.get("edges"), list):
        edge_items = json_data["edges"]
    elif isinstance(json_data.get("edges"), dict):
        edge_items = list(json_data["edges"].values())
    else:
        print("Error: 'edges' key is not a list or object/dict in JSON.", file=sys.stderr)
        return None

    # --- Graph Construction ---
    G = nx.DiGraph()
    node_ids = set()
    nodes_added, edges_added = 0, 0

    # Add nodes
    for i, node_data in enumerate(node_items):
        if not isinstance(node_data, dict) or "id" not in node_data or "label" not in node_data: continue
        node_id = str(node_data["id"]).strip(); node_label = str(node_data["label"]).strip()
        if not node_id or node_id in node_ids: continue
        G.add_node(node_id, label=node_label); node_ids.add(node_id); nodes_added += 1

    # Add edges
    for i, edge_data in enumerate(edge_items):
        if not isinstance(edge_data, dict) or "source" not in edge_data or "target" not in edge_data or "label" not in edge_data: continue
        source_id = str(edge_data["source"]).strip(); target_id = str(edge_data["target"]).strip(); edge_label = str(edge_data["label"]).strip()
        if not source_id or not target_id or source_id == target_id: continue
        # Only add edge if both nodes exist (important for visualization)
        if source_id in node_ids and target_id in node_ids:
            G.add_edge(source_id, target_id, label=edge_label); edges_added += 1
        else:
             print(f"  Skipping edge pointing to/from missing node: {source_id} -> {target_id}", file=sys.stderr)


    print(f"  Graph created for plotting: {nodes_added} nodes, {edges_added} edges.")
    return G

def plot_knowledge_graph(G: nx.DiGraph, title: str):
    """Plots a NetworkX graph attractively."""
    if not G or G.number_of_nodes() == 0:
        print(f"Skipping empty graph plot for: {title}")
        return

    num_nodes = G.number_of_nodes()
    print(f"  Attempting to plot graph with {num_nodes} nodes and {G.number_of_edges()} edges...")

    # Adjust figure size based on number of nodes (heuristic)
    figsize_base = 15
    figsize_factor = max(1, num_nodes / 50) # Increase size for larger graphs
    figsize = (int(figsize_base * figsize_factor), int(figsize_base * figsize_factor))
    # Cap the size to prevent excessively large figures
    max_fig_dim = 30
    figsize = (min(figsize[0], max_fig_dim), min(figsize[1], max_fig_dim))

    plt.figure(figsize=figsize)
    try:
        # Kamada-Kawai is often good for structure visualization
        print("  Calculating Kamada-Kawai layout...")
        pos = nx.kamada_kawai_layout(G)
        print("  Layout calculation finished.")
    except Exception as e_kk:
        print(f"  Kamada-Kawai layout failed ({e_kk}), trying spring layout...", file=sys.stderr)
        try:
            # Spring layout as fallback, needs more iterations for large graphs
            iterations = min(100, 50 + num_nodes // 2) # Heuristic for iterations
            print(f"  Calculating spring layout ({iterations} iterations)...")
            pos = nx.spring_layout(G, k=0.8/math.sqrt(num_nodes) if num_nodes > 0 else 0.8, # Adjust k based on node count
                                   iterations=iterations, seed=42)
            print("  Layout calculation finished.")
        except Exception as e_spring:
            print(f"  Spring layout also failed ({e_spring}), falling back to random layout.", file=sys.stderr)
            pos = nx.random_layout(G, seed=42)

    # Node styling
    node_size = max(30, 1500 / num_nodes) if num_nodes > 0 else 50 # Smaller nodes for larger graphs
    node_color = "lightblue"

    # Edge styling
    edge_color = "grey"
    edge_width = 0.6
    arrow_size = max(5, int(node_size / 4)) # Scale arrow size roughly with node size

    # Label styling
    font_size = max(5, 10 - num_nodes // 20) # Smaller font for larger graphs

    # Get node labels
    labels = nx.get_node_attributes(G, 'label')

    # Draw the graph elements
    print("  Drawing graph elements...")
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.9)

    # Draw edges with slight curve
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.4,
        edge_color=edge_color,
        width=edge_width,
        arrows=True,
        arrowstyle='-|>', # Simple arrow head
        arrowsize=arrow_size,
        connectionstyle='arc3,rad=0.05' # Slight curve
    )

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, font_color='#333333')

    plt.title(title, fontsize=max(10, font_size + 6)) # Adjust title size relative to labels
    plt.axis('off')
    plt.tight_layout() # Adjust layout to prevent overlap
    print("  Displaying plot...")
    plt.show()
    print("  Plot displayed.")


# --- Main Execution ---
if __name__ == "__main__":
    df_all, df_success = load_and_prepare_data(CSV_FILE)

    # 1. Plot Ranking per Concept
    print("\n--- Plotting Ranking per Concept ---")
    plot_ranking_per_concept(df_success)

    # 2. Plot Overall Ranking
    print("\n--- Plotting Overall Ranking ---")
    plot_overall_ranking(df_success)

    # 3. Find Top N Graphs and Plot Them
    print(f"\n--- Finding and Plotting Top {NUM_TOP_GRAPHS} Graphs by Average Degree ---")
    if df_success.empty:
        print("No successful runs to find top graphs.")
    else:
        top_graphs_info = df_success.sort_values('avg_average_degree', ascending=False).head(NUM_TOP_GRAPHS)

        if len(top_graphs_info) < NUM_TOP_GRAPHS:
             print(f"Warning: Found only {len(top_graphs_info)} successful runs, plotting those.")

        for index, row in top_graphs_info.iterrows():
            subject = row['subject']
            model = row['model']
            avg_degree = row['avg_average_degree']
            print(f"\nPlotting graph for: {subject} - {model} (Avg. Degree: {avg_degree:.4f})")

            # Construct expected filename (assuming retry 1 is representative)
            # You might want more sophisticated logic if retry 1 failed but others succeeded
            retry_num_to_load = 1 # Simplification: Load retry 1 JSON
            sanitized_subject = sanitize_filename(subject)
            sanitized_model = sanitize_filename(model)
            json_filename = f"{sanitized_subject}_{sanitized_model}_retry_{retry_num_to_load}.json"
            json_filepath = os.path.join(JSON_DIR, json_filename)

            if not os.path.exists(json_filepath):
                print(f"  Error: JSON file not found: {json_filepath}", file=sys.stderr)
                # Try finding *any* successful retry JSON for this combo? (More complex)
                continue

            try:
                with open(json_filepath, 'r', encoding='utf-8') as f_json:
                    # Use robust loading with potential cleaning (though less likely needed now)
                    content = f_json.read()
                    processed_content = content.strip()
                    if processed_content.startswith("```json") and processed_content.endswith("```"):
                        processed_content = processed_content[7:-3].strip()
                    elif processed_content.startswith("```") and processed_content.endswith("```"):
                        processed_content = processed_content[3:-3].strip()

                    first_brace = processed_content.find('{')
                    last_brace = processed_content.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                         potential_json_str = processed_content[first_brace : last_brace + 1]
                    else:
                         potential_json_str = processed_content # Fallback

                    if not potential_json_str.strip():
                         raise ValueError("JSON content is empty after cleaning/extraction")

                    json_data = json.loads(potential_json_str)

                # Create graph from loaded JSON data
                graph = create_graph_from_json(json_data)

                # Plot the graph
                if graph:
                    plot_title = f"Knowledge Graph: {subject}\nModel: {model}\n(Avg Degree: {avg_degree:.4f})"
                    plot_knowledge_graph(graph, plot_title)
                else:
                    print(f"  Failed to create graph object from JSON: {json_filepath}")

            except json.JSONDecodeError as e:
                print(f"  Error decoding JSON file {json_filepath}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"  An error occurred processing {json_filepath}: {e}", file=sys.stderr)

    print("\n--- Analysis Complete ---")