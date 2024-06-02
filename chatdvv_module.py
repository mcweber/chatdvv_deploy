# ---------------------------------------------------
# Version: 02.06.2024
# Author: M. Weber
# ---------------------------------------------------

from datetime import datetime
import os
from dotenv import load_dotenv
import re

import requests
import imaplib
import email
from bs4 import BeautifulSoup

# from validators import url as valid_url

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import openai
from groq import Groq
import ollama

import torch
from transformers import AutoTokenizer, AutoModel


# Init MongoDB Client
load_dotenv()
mongoClient = MongoClient(os.environ.get('MONGO_URI_DVV'))
database = mongoClient.dvv_content_pool
collection_artikel_pool = database.dvv_artikel_nahv

openaiClient = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY_DVV'))

groqClient = Groq(api_key=os.environ['GROQ_API_KEY_PRIVAT'])

# Load pre-trained model and tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Define Database functions ----------------------------------

def generate_abstracts(input_field: str, output_field: str, max_iterations: int = 20) -> None:

    cursor = collection_artikel_pool.find({output_field: ""}).limit(max_iterations)

    iteration = 0

    for record in cursor:

        iteration += 1

        start_time = datetime.now()
        abstract = write_summary(record[input_field])
        end_time = datetime.now()
        duration = end_time - start_time

        print(title[:50])
        print(f"#{iteration} Duration: {round(duration.total_seconds(), 2)}")
        print("-"*50)

        collection_artikel_pool.update_one({"_id": record.get('_id')}, {"$set": {output_field: abstract}})

    cursor.close()


def write_summary(text: str) -> str:

    if content == "":
        return "empty"

    systemPrompt = ["""
                    Du bist ein Redakteur im Bereich Transport und Verkehr.
                    Du bis Experte dafür, Zusammenfassungen von Fachartikeln zu schreiben.
                    Die maximale Länge der Zusammenfassungen sind 500 Wörter.
                    """]
    results = []
    history = []
    question = f"Erstelle eine Zusammenfassung des folgenden Textes: {text}. \
        Die Antwort darf nur aus dem eigentlichen Text der Zusammenfassung bestehen."
    summary = ask_llm("ollama_llama3", 0.1, question, history, systemPrompt, results)

    return summary


def generate_embeddings(input_field: str, output_field: str, 
                        max_iterations: int = 10) -> None:

    query = {input_field: {}, input_field: {"$ne": ""}}
    cursor = collection_artikel_pool.find(query)
    count = collection_artikel_pool.count_documents(query)
    print(f"Count: {count}")

    iteration = 0

    for record in cursor:

        iteration += 1
        if (max_iterations > 0) and (iteration > max_iterations):
            break

        summary_text = record.get(input_field)
        if summary_text is None:
            summary_text = "Keine Zusammenfassung vorhanden."

        embeddings = create_embeddings([summary_text])
        collection_artikel_pool.update_one({"_id": record.get('_id')}, {"$set": {output_field: embeddings}})


def create_embeddings(text: str) -> str:

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings_list = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    return embeddings_list


def ask_llm(llm: str, temperature: float = 0.2, question: str = "", history: list = [],
            systemPrompt: str = "", results_str: str = "") -> str:

    # define prompt
    input_messages = [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Hier sind einige relevante Informationen:\n" + results_str},
                {"role": "user", "content": "Basierend auf den oben genannten Informationen, " + question},
                ]

    if llm == "openai":
        response = openaiClient.chat.completions.create(
            model="gpt-4",
            temperature=temperature,
            messages = input_messages
            )
        output = response.choices[0].message.content

        # # Get the current usage balance
        # usage = openaiClient.Usage.retrieve()
        # print("Current usage balance:", usage['usage']['total'])

    elif llm == "groq":
        response = groqClient.chat.completions.create(
            model="mixtral-8x7b-32768",
            temperature=temperature,
            messages=input_messages
        )
        output = response.choices[0].message.content

    elif llm == "ollama_mistral":
        response = ollama.chat(model="mistral", temperature=temperature, messages=input_messages)
        output = response['message']['content']

    elif llm == "ollama_llama3":
        response = ollama.chat(model="llama3", messages=input_messages)
        output = response['message']['content']

    else:
        output = "Error: No valid LLM specified."

    return output


def text_search_artikel(search_text: str = "") -> [tuple, int]:

    if search_text != "":
        query = {"$text": {"$search": search_text }}
    else:
        query = {}

    fields = {"_id": 1, "titel": 1, "datum": 1, "untertitel": 1, "text_content": 1}
    # sort = [("datum", -1)]

    cursor = collection_artikel_pool.find(query, fields) #.sort(sort)
    count = collection_artikel_pool.count_documents(query)

    return cursor, count


def vector_search_artikel(query_string: str, limit: int = 10) -> tuple:

    embeddings = create_embeddings(query_string)

    pipeline = [
        {"$vectorSearch": {
            "index": "vector_index",
            "path": "embeddings",
            "queryVector": embeddings,
            "numCandidates": int(limit * 10),
            "limit": limit,
            }
        },
        {"$project": {
            "_id": 1,
            "titel": 1,
            "datum": 1,
            "untertitel": 1,
            "text": 1,
            "score": {"$meta": "vectorSearchScore"}
            }
        }
        ]

    result = collection_artikel_pool.aggregate(pipeline)

    return result

def print_results(cursor: list) -> None:

    if not cursor:
        print("Keine Artikel gefunden.")

    for i in cursor:
        print(f"[{str(i['datum'])[:10]}] {i['titel'][:70]}")
        # print("-"*80)
