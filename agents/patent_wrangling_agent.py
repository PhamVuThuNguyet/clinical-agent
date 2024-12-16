import os

import pandas as pd
import serpapi
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class GooglePatentsAgent:
    def __init__(self, api_key, query, output_file="extracted_data.csv", max_pages=10):
        self.client = serpapi.Client(api_key=api_key)
        self.query = query
        self.output_file = output_file
        self.max_pages = max_pages
        self.all_extracted_data = []  # Master data list
        self.page_number = 1  # Start from page 1

        self.name = "Google Patents Agent"
        self.role = """An Agent to use when looking for patents"""
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_patent_data",
                    "description": "Fetches Google Patents data for a given query and handles pagination.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for Google Patents.",
                            },
                            "max_pages": {
                                "type": "integer",
                                "description": "Maximum number of pages to scrape.",
                            },
                        },
                        "required": ["query", "max_pages"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_to_csv",
                    "description": "Saves extracted patent data to a CSV file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "output_file": {
                                "type": "string",
                                "description": "The name of the output CSV file.",
                            }
                        },
                        "required": ["output_file"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_summary",
                    "description": "Writes the full content of the CSV file to a plain text file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary_file": {
                                "type": "string",
                                "description": "The name of the output text file for the summary.",
                            }
                        },
                        "required": ["summary_file"],
                    },
                },
            },
        ]

    def fetch_patent_data(self):
        """
        Fetches patent data from Google Patents using SerpAPI.
        Handles pagination until max_pages or no more results.
        """
        while self.page_number <= self.max_pages:
            print(f"Fetching page {self.page_number} for query '{self.query}'...")
            # Perform search
            results = self.client.search(
                {
                    "engine": "google_patents",  # Google Patents engine
                    "q": self.query,  # Search query
                    "page": self.page_number,  # Current page number
                }
            )

            # Extract organic results
            organic_results = results.get("organic_results", [])
            if not organic_results:
                print("No more results found.")
                break

            # Process and store results
            for result in organic_results:
                self.all_extracted_data.append(
                    {
                        "title": result.get("title"),
                        "snippet": result.get("snippet"),
                        "filing_date": result.get("filing_date"),
                        "grant_date": result.get("grant_date"),
                        "inventor": result.get("inventor"),
                        "assignee": result.get("assignee"),
                        "patent_id": result.get("patent_id"),
                    }
                )

            # Check if more pages are available
            if "next" in results.get("serpapi_pagination", {}):
                self.page_number += 1
            else:
                break

    def save_to_csv(self):
        """
        Saves the extracted patent data into a CSV file.
        """
        if not self.all_extracted_data:
            print("No data to save.")
            return

        csv_columns = [
            "title",
            "snippet",
            "filing_date",
            "grant_date",
            "inventor",
            "assignee",
            "patent_id",
        ]

        pd.DataFrame(self.all_extracted_data).to_csv(
            self.output_file, columns=csv_columns, encoding="utf-8", index=False
        )
        print(f"Data saved to {self.output_file}")

    def write_summary(self, summary_file="summary.txt"):
        """
        Writes the full content of the CSV file to a plain text file.
        """
        if not os.path.exists(self.output_file):
            print(
                f"CSV file {self.output_file} does not exist. Run the agent to fetch data first."
            )
            return

        # Load the data from the CSV
        data = pd.read_csv(self.output_file)

        if data.empty:
            print("No data found in the CSV file.")
            return

        # Write the full content to a text file
        with open(summary_file, "w", encoding="utf-8") as f:
            for index, row in data.iterrows():
                f.write("Patent Record:\n")
                for column, value in row.items():
                    f.write(f"{column}: {value}\n")
                f.write("\n")

        print(f"Full content written to {summary_file}")

    def run(self):
        """
        Runs the agent: fetches patent data, saves it to a CSV file, and writes a summary.
        """
        print(f"Starting Google Patents Agent for query: {self.query}")
        self.fetch_patent_data()
        self.save_to_csv()
        self.write_summary()
        print("Processing completed.")


if __name__ == "__main__":
    # Ensure the .env file contains the SERPAPI_API_KEY
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        raise ValueError("Please set your SERPAPI_API_KEY in the .env file.")

    # Initialize the agent
    query = "parkinson"
    agent = GooglePatentsAgent(
        api_key=serpapi_api_key,
        query=query,
        output_file="extracted_data.csv",
        max_pages=10,
    )

    agent.run()
