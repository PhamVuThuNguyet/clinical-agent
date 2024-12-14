# ClinicalAgent

## Directory Structure

```
── agents/
│   │   ├── tools/
│   │   │   ├── drugbank/
│   │   │   ├── enrollment/
│   │   │   ├── hetionet/
│   │   │   ├── risk_model/
│   │   │   ├── graph_reasoning/
│   ├── main.ipynb
```

## Setup Instructions

### Setting the OpenAI API Key

Before starting, set your OpenAI API:

```sh
export OPENAI_API_KEY="sk-xxxxxxxxx"
```

### Dependencies

To set up the environment, run:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Agents and Tools

Before running ClinicalAgent, follow the README instructions in the `drugbank`, `enrollment`, `hetionet`, and `risk_model` directories to generate the necessary data for the tools:

- [drugbank]()
- [enrollment]()
- [hetionet]()
- [risk_model]()

