import os
from datetime import datetime

import pdfplumber
from openai import OpenAI
from PyPDF2 import PdfFileReader

from core.constants import LLMConstants

client = OpenAI(api_key=LLMConstants.OPENAI_API_KEY)


def extract_crucial_text(text):
    system_prompt = f"""
    You are an expert in Drug Development with extensive knowledge of Parkinson's disease (PD). Your objective is to meticulously extract and organize pivotal information from research articles on PD.
    Be comprehensiv but concise. Keep all the imprant details like biology, treatment pathway, relationships, reactions , molecular pathways, genetics. Only keep information connect to PD. Ignore information like author names, conflicts of interest, funding sources. Ignore references. Do not write summary for each page. 
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": text,
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def extract_text_from_pdfs(pdf_dir):
    text_data = {}
    key_text = {}

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)

            with pdfplumber.open(file_path) as pdf:
                text = ""
                ctext = ""

                for page in pdf.pages:
                    text += page.extract_text()
                    k_text = extract_crucial_text(page.extract_text())
                    ctext += k_text

                text_data[filename] = text
                key_text[filename] = ctext

    return [text_data, key_text]


if __name__ == "__main__":
    pdf_dir = "./TEXT_INPUT"
    output_dir = "./GRAPHDATA"
    [text_data, key_text] = extract_text_from_pdfs(pdf_dir)

    with open(f"{output_dir}/knowledgebasecomplete.txt", "w") as file:
        for filename, text in text_data.items():
            file.write(f"Filename: {filename}\n")
            file.write(f"Text: {text}\n\n")

    today_date = datetime.today().strftime("%Y-%m-%d")

    file_name = f"{output_dir}/knowledgebasecomplete_{today_date}.txt"
    with open(file_name, "w") as file:
        for filename, text in text_data.items():
            file.write(f"Filename: {filename}\n")
            file.write(f"Text: {text}\n\n")

    file_name = f"{output_dir}/knowledgebasecondensed_{today_date}.txt"
    with open(file_name, "w") as file:
        for filename, ctext in key_text.items():
            file.write(f"Filename: {filename}\n")
            file.write(f"Text: {ctext}\n\n")
            file.write(f"Text: {ctext}\n\n")
