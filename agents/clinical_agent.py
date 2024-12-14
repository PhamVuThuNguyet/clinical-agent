import re

from core.utils import LOGGER

from .efficiency_agent import EfficiencyAgent
from .enrollment_agent import EnrollmentAgent
from .graph_reasoning_agent import GraphReasoningAgent
from .LLMAgent import LLMAgent
from .reason_agent import decomposition
from .safety_agent import SafetyAgent


class ClinicalAgent(LLMAgent):
    def __init__(self, user_prompt, depth=1):
        self.user_prompt = user_prompt

        self.name = "clinical_agent"
        self.role = """You are a pharmaceutical research scientist and an expert in clinical trials"""

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "safety_agent",
                    "description": "The Safety Agent is designed to provide critical safety insights for pharmaceutical drugs. By consulting the Safety Agent, you can: 1. Retrieve historical failure rates of a given drug for a specific disease; 2. Assess the risk profile associated with the drug and the disease combination. Simply provide the drug name and disease name to access this valuable safety information.",
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
                    "name": "efficiency_agent",
                    "description": "The Efficiency Agent helps evaluate the effectiveness of a drug against specific diseases. Simply input the drug name and disease name to access comprehensive efficiency-related insights. It provides: 1. Drug introduction: Key details about the drug; 2. Disease introduction: Background information on the disease.; 3. Pathway analysis: The relationship and interaction between the drug and the disease within the Hetionet knowledge graph.",
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
                    "description": "The Enrollment Agent evaluates clinical trial eligibility criteria to predict the ease of patient enrollment. Use this agent to optimize eligibility criteria and ensure successful trial recruitment. By providing the eligibility criteria, the agent assesses and categorizes the enrollment potential into one of three levels: poor enrollment, good enrollment, or excellent enrollment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "eligibility_criteria": {
                                "type": "string",
                                "description": "eligibility criteria, contains including criteria and excluding criteria",
                            },
                            "drug_name": {
                                "type": "string",
                                "description": "The drug name",
                            },
                            "disease_name": {
                                "type": "string",
                                "description": "The disease name",
                            },
                        },
                        "required": [
                            "eligibility_criteria",
                            "drug_name",
                            "disease_name",
                        ],
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

        super().__init__(self.name, self.role, tools=self.tools, depth=depth)

    def safety_agent(self, drug_name, disease_name):
        LOGGER.log_with_depth(f"", depth=1)
        LOGGER.log_with_depth(f"Safety Agent...", depth=1)
        LOGGER.log_with_depth(f"Planing...", depth=1)
        LOGGER.log_with_depth(
            f"[Thought] Least to Most Reasoning: Decompose the original problem",
            depth=1,
        )

        safety_agent_ins = SafetyAgent(depth=2)

        decomposed_resp = decomposition(
            f"How can I evaluate the safety of the drug {drug_name} and disease {disease_name}?",
            tools=safety_agent_ins.tools,
        )

        subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", decomposed_resp)
        subproblems = [subproblem.strip() for subproblem in subproblems]

        for idx, subproblem in enumerate(subproblems):
            LOGGER.log_with_depth(f"<subproblem>{subproblem}</subproblem>", depth=1)

        LOGGER.log_with_depth(f"[Action] Solve each subproblem...", depth=1)
        problem_results = []
        for sub_problem in subproblems:
            LOGGER.log_with_depth(f"Solving...", depth=1)
            response = safety_agent_ins.request(
                f"The original user problem is: {self.user_prompt}\nNow, solve this problem: {sub_problem}"
            )

            if response == "":
                LOGGER.log_with_depth(
                    f"<solution>No solution found</solution>", depth=1
                )
                problem_results.append("No solution found")
            else:
                LOGGER.log_with_depth(f"<solution>{response}</solution>", depth=1)
                problem_results.append(response)

        return "\n".join(problem_results)

    def enrollment_agent(self, eligibility_criteria, drug_name, disease_name):
        LOGGER.log_with_depth(f"", depth=1)
        LOGGER.log_with_depth(f"Enrollment Agent...", depth=1)
        LOGGER.log_with_depth(f"Planing...", depth=1)
        LOGGER.log_with_depth(
            f"[Thought] Least to Most Reasoning: Decompose the original problem",
            depth=1,
        )

        enrollment_agent_ins = EnrollmentAgent(depth=2)

        response = enrollment_agent_ins.request(
            f"The original user problem is: {self.user_prompt}\nNow, evaluate the enrollment difficulty of the clinical trial with eligibility criteria: {eligibility_criteria}, drugs: {drug_name}, diseases: {disease_name}"
        )

        if response == "":
            LOGGER.log_with_depth(f"<solution>No solution found</solution>", depth=1)
            return "No solution found"
        else:
            LOGGER.log_with_depth(f"<solution>{response}</solution", depth=1)
            return response

    def efficiency_agent(self, drug_name, disease_name):
        LOGGER.log_with_depth(f"", depth=1)
        LOGGER.log_with_depth(f"Efficiency Agent...", depth=1)
        LOGGER.log_with_depth(f"Planing...", depth=1)
        LOGGER.log_with_depth(
            f"[Thought] Least to Most Reasoning: Decompose the original problem",
            depth=1,
        )

        efficiency_agent_ins = EfficiencyAgent(depth=2)

        decomposed_resp = decomposition(
            f"How can I evaluate the efficiency of the drug {drug_name} on the disease {disease_name}?",
            tools=efficiency_agent_ins.tools,
        )

        subproblems = re.findall(r"<subproblem>(.*?)</subproblem>", decomposed_resp)
        subproblems = [subproblem.strip() for subproblem in subproblems]

        for idx, subproblem in enumerate(subproblems):
            LOGGER.log_with_depth(f"<subproblem>{subproblem}</subproblem>", depth=1)

        LOGGER.log_with_depth(f"[Action] Solve each subproblem...", depth=1)

        problem_results = []
        for sub_problem in subproblems:
            response = efficiency_agent_ins.request(
                f"The original user problem is: {self.user_prompt}\nNow, solve this problem: {sub_problem}"
            )

            if response == "":
                LOGGER.log_with_depth(
                    f"<solution>No solution found</solution>", depth=1
                )
                problem_results.append("No solution found")
            else:
                LOGGER.log_with_depth(f"<solution>{response}</solution>", depth=1)
                problem_results.append(response)

        return "\n".join(problem_results)

    def graph_reasoning_agent(self, keyword_1, keyword_2, user_instruction):
        LOGGER.log_with_depth(f"", depth=1)
        LOGGER.log_with_depth(f"GraphReasoningAgent Agent...", depth=1)
        LOGGER.log_with_depth(f"Planing...", depth=1)
        LOGGER.log_with_depth(
            f"[Thought] Least to Most Reasoning: Decompose the original problem",
            depth=1,
        )

        graph_reasoning_agent_ins = GraphReasoningAgent(depth=2)

        response = graph_reasoning_agent_ins.request(
            f"""The user problem is: {self.user_prompt}\n. This is the user request: "{user_instruction}". How can I get the relevant context from graph knowledgebase about relationship between {keyword_1} and {keyword_2}?"""
        )

        if response == "":
            LOGGER.log_with_depth(f"<solution>No solution found</solution>", depth=1)
            return "No solution found"

        LOGGER.log_with_depth(f"<solution>{response}</solution>", depth=1)
        return response
