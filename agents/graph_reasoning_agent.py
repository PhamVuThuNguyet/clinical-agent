import os

import networkx as nx
from openai import OpenAI

from core.constants import LLMConstants

from .LLMAgent import LLMAgent
from .tools.graph_reasoning.GraphReasoning import (
    AutoModel,
    AutoTokenizer,
    find_path_and_reason,
    generate_node_embeddings,
    load_embeddings,
    make_graph_from_text,
    save_embeddings,
)

DATA_DIR = "./graph_reasoning/notebooks/GRAPHDATA"
TEXT_INPUT_DIR = "./graph_reasoning/notebooks/TEXT_INPUT/"
DATA_OUTPUT_DIR = "./graph_reasoning/notebooks/GRAPHDATA_OUTPUT/"
GRAPH_TEXT_FILE_NAME = "knowledgebasecondensed_2024-11-30.txt"
GRAPH_ROOT = "knowledgebasecondensed"

TOKENIZER_MODEL = "BAAI/bge-large-en-v1.5"

embedding_tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL,
)
embedding_model = AutoModel.from_pretrained(
    TOKENIZER_MODEL,
)
client = OpenAI(api_key=LLMConstants.OPENAI_API_KEY)


def complete_message_with_4o(system_prompt, prompt, temperature=0.333, max_tokens=4096):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


class GraphReasoningAgent(LLMAgent):
    def __init__(self, depth=1):
        self.name = "graph_reasoning_agent"
        self.role = """As an expert in graph programming, you have the capability 
        to answer to user's instruction by loading the graph and answer the user's instruction."""

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "answer_to_instruction",
                    "description": "To answer the user's instruction using information from a graph, ask the graph_reasoning Agent to answer the user's instruction. Given the graph root name, 2 keywords indicating 2 nodes or similar nodes in the graph, and the user's instruction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword_1": {
                                "type": "string",
                                "description": "The first keyword indicating a node in the graph",
                            },
                            "keyword_2": {
                                "type": "string",
                                "description": "The second keyword indicating a node in the graph",
                            },
                            "user_instruction": {
                                "type": "string",
                                "description": "The user's instruction to be answered",
                            },
                        },
                        "required": [
                            "keyword_1",
                            "keyword_2",
                            "user_instruction",
                        ],
                    },
                },
            }
        ]

        super().__init__(self.name, self.role, tools=self.tools, depth=depth)

    def answer_to_instruction(self, keyword_1, keyword_2, user_instruction):

        graph_name = f"{DATA_DIR}/{GRAPH_ROOT}_graphML.graphml"

        if os.path.exists(graph_name):
            G = nx.read_graphml(graph_name)

        else:
            with open(os.path.join(TEXT_INPUT_DIR, GRAPH_TEXT_FILE_NAME), "r") as f:
                text = f.read()
                graph_HTML, graph_GraphML, G, net, _ = make_graph_from_text(
                    txt=text,
                    graph_root=GRAPH_ROOT,
                    generate=complete_message_with_4o,
                    data_dir=DATA_DIR,
                    chunk_size=2500,
                )

        embedding_file = f"{GRAPH_ROOT}_embeddings_ge-large-en-v1.5.pkl"
        embedding_path = f"{DATA_DIR}/{embedding_file}"

        if os.path.exists(embedding_path):
            node_embeddings = load_embeddings(embedding_path)

        else:
            node_embeddings = generate_node_embeddings(
                G,
                embedding_tokenizer,
                embedding_model,
            )
            save_embeddings(node_embeddings, embedding_path)

        (
            response,
            (best_node_1, best_similarity_1, best_node_2, best_similarity_2),
            path,
            path_graph,
            shortest_path_length,
            fname,
            graph_GraphML,
        ) = find_path_and_reason(
            G,
            node_embeddings,
            embedding_tokenizer,
            embedding_model,
            complete_message_with_4o,
            data_dir=DATA_OUTPUT_DIR,
            verbatim=True,
            include_keywords_as_nodes=True,  # Include keywords in the graph analysis
            keyword_1=keyword_1,  # "parkinson's disease",
            keyword_2=keyword_2,  # "treatment groups",
            N_limit=9999,  # The limit for keywords, triplets, etc.
            instruction=user_instruction,  # 'Develop a new research idea on a new drug that can help mitigate Parkinson desease.',
            keywords_separator=", ",
            graph_analysis_type="nodes and relations",
            temperature=0.3,
            inst_prepend="### ",  # Instruction prepend text
            prepend="""You are given a set of information from a graph that describes the relationship 
                    between materials, structure, properties, and properties. You analyze these logically 
                    through reasoning.\n\n""",  # Prepend text for analysis
            visualize_paths_as_graph=True,  # Whether to visualize paths as a graph
            display_graph=True,  # Whether to display the graph
        )
        with open(os.path.join(DATA_OUTPUT_DIR, f"{GRAPH_ROOT}_output.txt"), "w") as f:
            f.write(response)
        return response
