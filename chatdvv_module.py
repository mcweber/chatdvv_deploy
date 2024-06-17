# ---------------------------------------------------
# Version: 17.06.2024
# Author: M. Weber
# ---------------------------------------------------
# 07.06.2024 Adapted fulltext search to atlas search
# 13.06.2024 Added get_document()
# 15.06.2024 Added reset filter. Added filter for textsearch
# 15.06.2024 Added torch model for embeddings
# 15.06.2024 Updated vector_search to use text_embeddings
# 17.06.2024 Vektor search only without filter
# ---------------------------------------------------

from datetime import datetime
import os
from dotenv import load_dotenv

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
collection = database.dvv_artikel
openaiClient = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY_DVV'))
groqClient = Groq(api_key=os.environ['GROQ_API_KEY_PRIVAT'])

# Load pre-trained model and tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define Database functions ----------------------------------

def generate_abstracts(input_field: str, output_field: str, max_iterations: int = 20) -> None:
    cursor = collection.find({output_field: ""}).limit(max_iterations)
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
        collection.update_one({"_id": record.get('_id')}, {"$set": {output_field: abstract}})
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
    cursor = collection.find({output_field: {}})
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
            collection.update_one({"_id": record['_id']}, {"$set": {output_field: embeddings}})
    print(f"\nGenerated embeddings for {iteration} records.")


# def create_embeddings(text: str) -> str:
#     if text == "":
#         return "empty"
#     model = "text-embedding-3-small"
#     text = text.replace("\n", " ")
#     return openaiClient.embeddings.create(input = [text], model=model).data[0].embedding


def create_embeddings(text: str) -> list:
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


def generate_filter(filter: list, field: str) -> dict:
    if filter:
        return {field: {"$in": filter}}
    else:
        return {}
    

def text_search(search_text : str = "*", filter : list = [], limit : int = 10) -> [tuple, int]:
    # query = {"$search": {"index": "volltext", "text": search_text, "path": {"wildcard": "*"}}}
    query = {
        "index": "volltext",
        "sort": {"date": -1},
        "text": {
            "query": search_text, 
            "path": {"wildcard": "*"}
            }
        }
    fields = {
        "_id": 1,
        "quelle_id": 1,
        "jahrgang": 1,
        "nummer": 1,
        "titel": 1, 
        "datum": 1,
        "date": 1,
        "untertitel": 1, 
        "text": 1, 
        "ki_abstract": 1
        }
    pipeline = [
        {"$search": query},
        {"$match": {"quelle_id": {"$in": filter}}},
        # {"$match": generate_filter(filter, "quelle_id")},
        {"$project": fields},
        {"$limit": limit}
        ]
    cursor = collection.aggregate(pipeline)
    # count = collection.aggregate(pipeline_meta)
    count = 0
    return cursor, count


def vector_search(query_string: str = "", filter : list = [], sort: str = "date", limit: int = 10) -> tuple:
    embeddings_query = create_embeddings(query_string)
    query = {
            "index": "text_vector_index",
            "path": "text_embeddings",
            "queryVector": embeddings_query,
            "numCandidates": int(limit * 10),
            "limit": limit,
            # "filter": {"quelle_id_embeddings": "THB"}
            }
    fields = {
            "_id": 1,
            "quelle_id": 1,
            "jahrgang": 1,
            "nummer": 1,
            "titel": 1,
            "datum": 1,
            "untertitel": 1,
            "text": 1,
            "ki_abstract": 1,
            "date": 1,
            "score": {"$meta": "vectorSearchScore"}
            }
    pipeline = [
        {"$vectorSearch": query},
        {"$sort": {sort: -1}},
        {"$project": fields}
        ]
    return collection.aggregate(pipeline)


def print_results(cursor: list) -> None:
    if not cursor:
        print("Keine Artikel gefunden.")
    for item in cursor:
        print(f"[{str(item['datum'])[:10]}] {item['titel'][:70]}")
        # print("-"*80)


def group_by_field() -> dict:
    pipeline = [
            {
            '$group': {
                '_id': '$quelle_id', 
                'count': {
                    '$sum': 1
                    }
                }
            }, {
            '$sort': {
                'count': -1
                }
            }
            ]
    result = collection.aggregate(pipeline)
    # transfor into dict
    return_dict = {}
    for item in result:
        return_dict[item['_id']] = item['count']
    return return_dict


def list_fields() -> dict:
    result = collection.find_one()
    return result.keys()


def get_document(id: str) -> dict:
    document = collection.find_one({"id": id})
    return document