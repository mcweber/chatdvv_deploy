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

# Init MongoDB Client
load_dotenv()
mongoClient = MongoClient(os.environ.get('MONGO_URI_DVV'))
database = mongoClient.dvv_content_pool
collection_artikel_pool = database.dvv_artikel_nahv

openaiClient = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY_DVV'))

groqClient = Groq(api_key=os.environ['GROQ_API_KEY_PRIVAT'])

# Define Database functions ----------------------------------

def generate_abstracts(input_field: str, output_field: str, max_iterations: int = 20) -> None:

    cursor = collection_artikel_pool.find({output_field: ""}).limit(max_iterations)

    iteration = 0

    for record in cursor:

        iteration += 1

        start_time = datetime.now()
        abstract = write_summary(str(record[input_field]))
        end_time = datetime.now()
        duration = end_time - start_time

        print(record['titel'][:50])
        print(f"#{iteration} Duration: {round(duration.total_seconds(), 2)}")
        print("-"*50)

        collection_artikel_pool.update_one({"_id": record.get('_id')}, {"$set": {output_field: abstract}})

    cursor.close()


def write_summary(text: str) -> str:

    if text == "":
        return "empty"

    systemPrompt = """
                    Du bist ein Redakteur im Bereich Transport und Verkehr.
                    Du bis Experte dafür, Zusammenfassungen von Fachartikeln zu schreiben.
                    Die maximale Länge der Zusammenfassungen sind 500 Wörter.
                    Wichtig ist nicht die Lesbarkeit, sondern die Kürze und Prägnanz der Zusammenfassung:
                    Was sind die wichtigsten Aussagen und Informationen des Textes?
                    """
    task = """
            Erstelle eine Zusammenfassung des Originaltextes in deutscher Sprache.
            Verwende keine Zeilenumrüche oder Absätze.
            Die Antwort darf nur aus dem eigentlichen Text der Zusammenfassung bestehen.
            """
   
    prompt = [
            {"role": "system", "content": systemPrompt},
            {"role": "assistant", "content": f'Originaltext: {text}'},
            {"role": "user", "content": task}
            ]
    
    response = openaiClient.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages = prompt
            )
    
    return response.choices[0].message.content


def generate_embeddings(input_field: str, output_field: str, 
                        max_iterations: int = 10) -> None:

    query = {output_field: {}}
    cursor = collection_artikel_pool.find(query)
    count = collection_artikel_pool.count_documents(query)
    print(f"Records without embeddings: {count}")

    iteration = 0

    for record in cursor:

        iteration += 1
        if iteration > max_iterations:
            break

        article_text = record[input_field]
        if article_text == "":
            article_text = "Fehler: Kein Text vorhanden."
        else:
            embeddings = create_embeddings(article_text)
            collection_artikel_pool.update_one({"_id": record['_id']}, {"$set": {output_field: embeddings}})

    print(f"\nGenerated embeddings for {iteration} records.")


def create_embeddings(text: str) -> str:

    model = "text-embedding-3-small"
    text = text.replace("\n", " ")
    return openaiClient.embeddings.create(input = [text], model=model).data[0].embedding


def ask_llm(llm: str, temperature: float = 0.2, question: str = "", history: list = [],
            systemPrompt: str = "", results_str: str = "") -> str:

    # define prompt
    input_messages = [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": 'Hier sind einige relevante Informationen:\n'  + results_str},
                {"role": "user", "content": 'Basierend auf den oben genannten Informationen, ' + question}
                ]

    if llm == "openai":
        response = openaiClient.chat.completions.create(
            model="gpt-4o",
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
        response = ollama.chat(model="mistral", messages=input_messages)
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

    fields = {"_id": 1, "titel": 1, "datum": 1, "untertitel": 1, "text": 1, "ki_abstract": 1}
    # sort = [("datum", -1)]

    cursor = collection_artikel_pool.find(query, fields) #.sort(sort)
    count = collection_artikel_pool.count_documents(query)

    return cursor, count


def vector_search_artikel(query_string: str, limit: int = 10) -> tuple:

    embeddings_query = create_embeddings(query_string)

    pipeline = [
        {"$vectorSearch": {
            "index": "nahv_vector_index",
            "path": "embeddings",
            "queryVector": embeddings_query,
            "numCandidates": int(limit * 10),
            "limit": limit,
            }
        },
        {"$project": {
            "_id": 1,
            "quelle_id": 1,
            "jahrgang": 1,
            "nummer": 1,
            "titel": 1,
            "datum": 1,
            "untertitel": 1,
            "text": 1,
            "ki_abstract": 1,
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
