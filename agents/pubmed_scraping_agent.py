import asyncio
import os
import re
from datetime import datetime

import aiohttp
import nest_asyncio
import requests
from Bio import Entrez


class PubMedAgent:
    def __init__(self, email, download_folder="downloads"):
        self.email = email
        self.download_folder = download_folder
        Entrez.email = email
        os.makedirs(download_folder, exist_ok=True)
        self.name = "PubMed Agent"
        self.role = """An expert PubMed Agent for searching articles, fetching metadata, and downloading PDFs."""
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "pdf_url_from_doi",
                    "description": "Given a DOI, retrieves the PDF URL using the OpenAlex API.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doi": {
                                "type": "string",
                                "description": "The DOI of the article.",
                            }
                        },
                        "required": ["doi"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_safe_filename",
                    "description": "Creates a sanitized filename from article metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metadata": {
                                "type": "object",
                                "description": "Metadata of the article, including title, authors, and date.",
                            }
                        },
                        "required": ["metadata"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_safe_foldername",
                    "description": "Generates a sanitized folder name from a keyword query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword_query": {
                                "type": "string",
                                "description": "The keyword query to generate the folder name.",
                            }
                        },
                        "required": ["keyword_query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_and_fetch_pubmed",
                    "description": "Searches PubMed for articles matching the keyword and author queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword_query": {
                                "type": "string",
                                "description": "The keyword query for searching PubMed.",
                            },
                            "author_query": {
                                "type": "string",
                                "description": "The author query for searching PubMed.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to retrieve.",
                            },
                        },
                        "required": ["keyword_query", "max_results"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_article_metadata",
                    "description": "Extracts and sanitizes metadata from a PubMed article.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "article": {
                                "type": "object",
                                "description": "The PubMed article object to extract metadata from.",
                            }
                        },
                        "required": ["article"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "download_pdf",
                    "description": "Downloads the PDF of a PubMed article if available.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session": {
                                "type": "object",
                                "description": "The aiohttp session used for downloading.",
                            },
                            "article": {
                                "type": "object",
                                "description": "The PubMed article object to download the PDF for.",
                            },
                            "folder": {
                                "type": "string",
                                "description": "The folder to save the downloaded PDF.",
                            },
                        },
                        "required": ["session", "article", "folder"],
                    },
                },
            },
        ]

    def pdf_url_from_doi(self, doi):
        api_res = requests.get(f"https://api.openalex.org/works/https://doi.org/{doi}")
        # Raise an exception for bad status codes to be handled by the caller
        api_res.raise_for_status()
        metadata = api_res.json()
        pdf_url = metadata.get("open_access", {}).get("oa_url")
        if pdf_url is None:
            if metadata.get("host_venue"):
                pdf_url = metadata["host_venue"].get("url")
            elif metadata.get("primary_location"):
                pdf_url = metadata["primary_location"].get("landing_page_url")
        if not pdf_url:
            print(f"No PDF found for DOI: {doi}")
        return pdf_url

    def create_safe_filename(self, metadata):
        first_author = (
            metadata["authors"][0].split()[0] if metadata["authors"] else "Unknown"
        )
        year = metadata["datetime"].year
        title_part = (
            re.sub(r"[^\w\s-]", "", metadata["title"])[:50].strip().replace(" ", "_")
        )
        filename = f"{first_author}_{year}_{title_part}.pdf"
        return re.sub(r"[^\w\.-]", "_", filename)

    def create_safe_foldername(self, keyword_query):
        folder_name = re.sub(r"[^\w\-_\. ]", "_", keyword_query)
        return folder_name.replace(" ", "_")

    async def search_and_fetch_pubmed(
        self, keyword_query, author_query=None, max_results=20
    ):
        query_parts = []
        if keyword_query:
            query_parts.append(f"({keyword_query})")
        if author_query:
            query_parts.append(f"({author_query}[Author])")
        if not query_parts:
            raise ValueError(
                "At least one of keyword_query or author_query must be provided"
            )

        full_query = " AND ".join(query_parts)
        handle = Entrez.esearch(
            db="pubmed", term=full_query, retmax=max_results, sort="relevance"
        )
        record = Entrez.read(handle)
        ids = record["IdList"]

        if not ids:
            return [], record["QueryTranslation"]

        handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
        articles = Entrez.read(handle)["PubmedArticle"]
        return articles, record["QueryTranslation"]

    def fetch_article_metadata(self, article):
        title = re.sub(
            "<[^<]+?>", "", article["MedlineCitation"]["Article"]["ArticleTitle"]
        )
        metadata = {
            "pmid": article["MedlineCitation"]["PMID"],
            "title": title,
            "journal": article["MedlineCitation"]["Article"]["Journal"]["Title"],
            "date": article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"][
                "PubDate"
            ],
            "authors": [
                author.get("LastName", "") + " " + author.get("Initials", "")
                for author in article["MedlineCitation"]["Article"]["AuthorList"]
            ],
        }
        year = metadata["date"].get("Year", "1900")
        month = metadata["date"].get("Month", "1")
        day = metadata["date"].get("Day", "1")

        month_dict = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }
        if month in month_dict:
            month = month_dict[month]
        metadata["datetime"] = datetime(int(year), int(month), int(day))
        print(metadata)
        return metadata

    async def download_pdf(self, session, article, folder):
        metadata = self.fetch_article_metadata(article)
        filename = self.create_safe_filename(metadata)
        file_path = os.path.join(folder, filename)

        if os.path.exists(file_path):
            print(f"Skipping download: {filename} already exists")
            return metadata

        article_id = article["PubmedData"]["ArticleIdList"]
        pmc_id = next(
            (id for id in article_id if id.attributes["IdType"] == "pmc"), None
        )
        doi = next((id for id in article_id if id.attributes["IdType"] == "doi"), None)

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/pdf",
        }

        if pmc_id:
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf"
            print(f"Downloading PDF from {pdf_url}")
        elif doi:
            pdf_url = self.pdf_url_from_doi(doi)
            print(f"Downloading PDF from {pdf_url}")
        else:
            print(f"No direct PDF link available for {metadata['pmid']}")
            return None

        async with session.get(pdf_url, headers=headers) as response:
            if response.status == 200:
                content = await response.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                print(f"Downloaded: {file_path}")
                return metadata
            else:
                print(
                    f"Failed to download {metadata['title']} from {pdf_url}. Status code: {response.status}"
                )
        return None

    async def main(self, search_query, max_results):
        folder_name = self.create_safe_foldername(search_query)
        folder_path = os.path.join(self.download_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        articles, query_translation = await self.search_and_fetch_pubmed(
            search_query, max_results=max_results
        )

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.download_pdf(session, article, folder_path) for article in articles
            ]
            results = await asyncio.gather(*tasks)

        summary_file = os.path.join(folder_path, "summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            for result in results:
                if result:
                    f.write(f"Title: {result['title']}\n")
                    f.write(f"Authors: {', '.join(result['authors'])}\n")
                    f.write(f"Journal: {result['journal']}\n")
                    f.write(f"Date: {result['datetime'].strftime('%Y-%m-%d')}\n")
                    f.write(f"PMID: {result['pmid']}\n\n")

        print(f"Download complete. Check the '{folder_path}' folder.")


if __name__ == "__main__":

    nest_asyncio.apply()

    email = "your_email@example.com"
    agent = PubMedAgent(email=email)
    search_query = "Parkinson's disease [Title][Abstract]"
    max_results = 20
    asyncio.run(agent.main(search_query=search_query, max_results=max_results))
