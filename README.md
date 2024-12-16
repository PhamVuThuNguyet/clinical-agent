# ClinicalAgent: AI-Powered Clinical Trial Analysis System

ClinicalAgent is an advanced system that leverages multiple specialized AI agents to analyze and predict clinical trial outcomes. The system uses a modular architecture with different agents working together to provide comprehensive insights about drug safety, efficiency, and trial feasibility.

## System Architecture

![alt text](<Clinical Agent.jpg>)

## Key Components

### 1. LLMAgent (Base Agent)
- Handles communication with language models
- Manages tool calling and response processing

### 2. Planning Agent & Reasoning Agent

The Planning Agent’s primary role is to strategize and determine the optimal approach to address user problems. Utilizing the **LEAST-TO-MOST Reasoning method**, this agent systematically decomposes complex issues into smaller, more manageable subproblems. This stepwise breakdown facilitates targeted interventions, where each subproblem is addressed by the most suitable specialist agent. Subsequently, within this structured framework, **ReAct reasoning** is applied to each segment. This involves recognizing relevant patterns or cues, deciding on appropriate actions based on these insights, and adapting these actions by considering the immediate context.

### 3. Specialized Agents

#### Enrollment Agent
Proper enrollment ensures that the trial has enough participants to statistically power the study. The Enrollment Agent is a hierarchical transformer-based model that integrates sentence embeddings from a fine-tuned BioBERT. It takes eligibility criteria as an input feature and predicts the enrollment success rate. This is formulated as a binary classification problem, where 1 denotes successful enrollment and 0 denotes unsuccessful enrollment.

#### Efficiency Agent
The Efficacy Agent is primarily focused on assessing the therapeutic effectiveness of drugs against specified diseases. Specifically, it employs the **SMILES (Simplified Molecular Input Line Entry System) notation** to identify and retrieve detailed chemical and pharmacological information about drugs. This includes their molecular structure, mechanism of action, metabolism, and potential side effects, providing a holistic view of the drug’s properties.

#### Safety Agent
The Safety Agent is focused on the assessment of drug safety and its implications for patient health. This agent leverages a comprehensive repository of pharmacological data and historical clinical trial outcomes to evaluate the risks associated with specific drug-disease interactions.

### 3. Data Aggregation Agents
#### Pubmed Agent
The PubMed Agent is designed to retrieve, process, and analyze biomedical literature from the PubMed database. By leveraging advanced NLP techniques, this agent identifies relevant publications based on user queries, extracts key information, and summarizes findings. 

#### Google Patent Agent
The Google Patent Agent specializes in wrangling related patent data.

### 4. RAG & Reasoning Agents
#### Drugbank Database Retrieval Agent
This agent will be used to retrieve up-to-date, detailed descriptions of the drug and the disease from DrugBank.

#### Hetionet & Custom Graph Retrieval Agent
This agent utilizes the HetioNet and Custom Knowledge Graph to trace and visualize the pathways connecting the drug to the disease. This involves identifying biological interactions, such as target proteins and genetic associations, that are crucial for understanding the drug’s potential efficacy.

## Data sources
- Drugbank
- Hetionet
- ClinicalTrials.gov
- Pubmed
- Google Patents

## Installation
1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up the environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export SERPAPI_API_KEY="your-serpapi-key"
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

