import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from agents import clinical_agent
from agents.planning_agent import decomposition
from core.utils import LOGGER, llm_request

client = OpenAI()

sys.path.append(os.getcwd())
load_dotenv()


def solve_problem(user_problem):
    agent_tools = [
        {
            "type": "function",
            "function": {
                "name": "safety_agent",
                "description": "To understand the safety of the drug, including toxicity and side effects, consult the Safety Agent for safety information. Given drug name, return the safety information of the drug, e.g. ADMET, drug introduction, toxicity and side effects etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "The drug name",
                        }
                    },
                    "required": ["drug_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "efficiency_agent",
                "description": "To assess the drug's efficiency against the diseases, ask the efficiency Agent for information regarding the drug's effectiveness on the disease. Given drug name and disease name, return the drug introduction, disease introduction, and the path between drug and disease in the hetionet knowledge graph etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "The drug name",
                        },
                        "disease_name": {
                            "type": "string",
                            "description": "The disease name",
                        },
                    },
                    "required": ["drug_name", "disease_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "enrollment_agent",
                "description": "To determine if the clinical trial's eligibility criteria facilitate easy enrollment with enough patients. Given eligibility criteria, return the clinical trial will be poor enrollment, good enrollment, or excellent enrollment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "eligibility_criteria": {
                            "type": "string",
                            "description": "eligibility criteria, contains including criteria and excluding criteria",
                        }
                    },
                    "required": ["eligibility_criteria"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_reasoning_agent",
                "description": "The Graph Reasoning Agent is designed to interpret and respond to requests by analyzing graph structures. The agent loads the specified graph, processes the relationships or paths between the nodes, and delivers a response aligned with the user's instruction. Ideal for tasks involving graph-based data reasoning and exploration.",
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
                            "description": "The request to be answered",
                        },
                    },
                    "required": [
                        "keyword_1",
                        "keyword_2",
                        "user_instruction",
                    ],
                },
            },
        },
    ]

    LOGGER.log_with_depth(f"Decomposing the problem...")
    decomposed_resp = decomposition(user_problem, agent_tools)

    subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", decomposed_resp)
    subproblems = [subproblem.strip() for subproblem in subproblems]

    for idx, subproblem in enumerate(subproblems):
        LOGGER.log_with_depth(f"[PROBLEM]: {subproblem}")

    problem_results = []

    clinicalAgent = clinical_agent.ClinicalAgent(user_problem)

    for sub_problem in subproblems:
        LOGGER.log_with_depth(f"\t[PROBLEM]: {sub_problem}...")
        response = clinicalAgent.request(
            f"The original user problem is: {user_problem}\nNow, solve this problem: {sub_problem}"
        )

        LOGGER.log_with_depth(f"\t[SOLUTION]: {response}\n")
        problem_results.append(response)

    messages = []
    fewshot_examples = open("few_shot.txt", "r").read()
    system_prompt = f""" 
        You are an expert in clinical trials. Based on the subproblems have solved, please solve the user's problem and provide the reason.
        Firstly, please give the final result of the user's problem, you must give a specific results, clear answer the user's problem.
        Secondly, please provide the reason step by step for the result.
        You must include the exact probability within the <final_result></final_result> tags, e.g., 'The failure rate of the clinical trial is <final_result>0.8</final_result>.'
        Third, explain the reason step by step.
        Noted, you must include the exact probability within the <final_result></final_result> tags.

        The following examples are essential for your understanding, offering significant learning opportunities. By analyzing these examples, including sub-problems and labels, you will learn how to make accurate predictions.
        Each example is within the <example></example> tags.
        Examples:
        {fewshot_examples}
    """

    messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {"role": "user", "content": f"The original user problem is: {user_problem}"}
    )
    messages.append(
        {
            "role": "user",
            "content": f"The subproblems have solved are: {problem_results}",
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "Please solve the user's problem and provide the reason.",
        }
    )

    final_results = llm_request(messages)
    LOGGER.log_with_depth(f"Final results:\n")
    LOGGER.log_with_depth(final_results.choices[0].message.content)
    LOGGER.log_with_depth("\n===============================================\n\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        LOGGER.log_with_depth("Error: Please provide the random_idx argument.")
        sys.exit(1)

    LOGGER.log_with_depth(f"Random Index: {sys.argv[1]}")
    random_idx = int(sys.argv[1])

    cwd_path = os.getcwd()
    trial_df = pd.read_csv(
        f"{cwd_path}/agents/tools/risk_model/data/trial_success.csv", sep="\t"
    )
    trial_row = trial_df.iloc[random_idx]

    try:
        nctid = trial_row["nctid"]
        criteria = trial_row["criteria"]
        drugs = trial_row["drugs"]
        diseases = trial_row["diseases"]
        label = trial_row["label"]

        user_problem = f"""
        I have designed a clinical trial and hope you can help me predict whether this trial can pass.
        #criteria#: {criteria}
        #drugs#: {drugs}
        #diseases#: {diseases}
        """

        LOGGER.log_with_depth(
            f"NCTID: {nctid}\nUser problem:\n {user_problem}\n\n Correct Label: {label}, 1 means passed, 0 means not passed.\n"
        )

        solve_problem(user_problem)

        LOGGER.log_with_depth("\n\n\n\n\n\n\n\n")
    except Exception as e:
        LOGGER.log_with_depth(f"Error: {e}")
