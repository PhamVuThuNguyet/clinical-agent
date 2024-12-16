# Graph Reasoning Agent

The Graph Reasoning Agent is a specialized component of the ClinicalAgent system that performs sophisticated analysis and reasoning over biomedical knowledge graphs. It enables complex queries about relationships between drugs, diseases, and other biomedical entities.

## Overview

The Graph Reasoning Agent leverages graph structures to:
- Find and analyze paths between biomedical entities
- Reason about relationships in the knowledge graph
- Generate explanations for discovered connections
- Support complex queries about drug-disease relationships

## Features

### 1. Path Analysis
- Finds paths between specified nodes in the knowledge graph
- Analyzes relationship types and connection strengths
- Supports multiple path lengths and relationship types

### 2. Embedding-Based Reasoning
- Uses BAAI/bge-large-en-v1.5 for node embeddings
- Enables semantic similarity searches
- Supports finding similar entities and relationships

### 3. Knowledge Integration
- Integrates with the Hetionet Knowledge Graph
- Supports custom knowledge graph inputs
- Maintains graph structure in memory for fast querying

## Getting Started
### Step 1. Prepare textual information

- The textual information should be cleaned before building the graph

- Put the textual information under a ".txt" file in the ```agents/tools/graph_reasoning/notebooks/TEXT_INPUT``` folder

### Step 2. Build the graph using the textual information
There are 2 methods to build a graph:
- When calling the graph_reasoning_agent, provide the graph_text_file_name as the file prepared in ```Step 1``` and provide a unique graph_root for the new graph

- Edit the ```GRAPH_TEXT_FILE_NAME``` and ```GRAPH_ROOT``` directly from ```agents/graph_reasoning_agent.py```  

The Agent will then load the text file and create a new graph.

Note: If the graph_root is an existing graph, the Agent will not create a new graph but load existing one.

### Step 3. Visualize the graph

- Once the graph building process is completed, go to the ```agents/tools/graph_reasoning/notebooks/GRAPHDATA``` folder
- Locate the HTML file of the newly created graph. It should have the following format: <graph_root>_grapHTML.html
- Open the HTML file in your browser and play with the graph