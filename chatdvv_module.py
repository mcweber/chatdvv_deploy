# ---------------------------------------------------
# Version: 24.07.2024
# Author: M. Weber
# ---------------------------------------------------
# 07.06.2024 Adapted fulltext search to atlas search
# 13.06.2024 Added get_document()
# 15.06.2024 Added reset filter. Added filter for textsearch
# 15.06.2024 Added torch model for embeddings
# 15.06.2024 Updated vector_search to use text_embeddings
# 21.06.2024 added update_systemprompt and get_systemprompt
# 21.06.2024 added more LLMs
# 22.06.2024 added anthropic
# 22.06.2024 added web search with duckduckgo
# 23.06.2024 switched web search to duckduckgo_search
# 26.06.2024 added current date to system prompt
# 26.06.2024 switched websearch to news-search
# 06.07.2024 addes scores-sorting in text_search and vector_search
# 06.07.2024 added tavily web search
# 07.07.2024 added write_takeaways, generate_keywords
# 08.07.2024 added score threshold in search functions (in progress)
# 12.07.2024 added wildcard option to text_search
# 13.07.2024 added industry filter to vector_search
# 24.07.2024 added LLAMA 3.1 70B and GPT-4o-mini
# ---------------------------------------------------

from datetime import datetime
import os
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import openai
import anthropic
from groq import Groq
import ollama

from duckduckgo_search import DDGS
from tavily import TavilyClient

import torch
# from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel

# Define global variables ----------------------------------
LLMS = ("GPT 4o mini", "GPT 4o", "anthropic", "groq_mixtral-8x7b-32768", "groq_llama-3.1-70b", "groq_gemma-7b-it")

# Init MongoDB Client
load_dotenv()
mongoClient = MongoClient(os.environ.get('MONGO_URI_DVV'))
database = mongoClient.dvv_content_pool
collection = database.dvv_artikel
collection_config = database.config
openaiClient = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY_DVV'))
anthropicClient = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY_DVV'))
groqClient = Groq(api_key=os.environ['GROQ_API_KEY_PRIVAT'])
tavilyClient = TavilyClient(api_key=os.environ['TAVILY_API_KEY_DVV'])

# Load pre-trained model and tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "bert-base-german-cased" # 768 dimensions
# model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

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

def write_summary(text: str = "", length: int = 500) -> str:
    if text == "":
        return "empty"
    systemPrompt = f"""
                    Du bist ein Redakteur im Bereich Transport und Verkehr.
                    Du bis Experte dafür, Zusammenfassungen von Fachartikeln zu schreiben.
                    Die maximale Länge der Zusammenfassungen sind {length} Wörter.
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

def write_takeaways(text: str = "", max_takeaways: int = 5) -> str:
    if text == "":
        return "empty"
    systemPrompt = """
                    Du bist ein Redakteur im Bereich Transport und Verkehr.
                    Du bis Experte dafür, die wichtigsten Aussagen von Fachartikeln herauszuarbeiten.
                    """
    task = f"""
            Erstelle eine Liste der wichtigsten Aussagen des Textes in deutscher Sprache.
            Es sollten maximal {max_takeaways} Aussagen sein.
            Jede Aussage sollte kurz und prägnant in einem eigenen Satz formuliert sein.
            Die Antwort darf nur aus den eigentlichen Aussagen bestehen.
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

def generate_keywords(text: str = "", max_keywords: int = 5) -> str:
    if text == "":
        return "empty"
    systemPrompt = """
                    Du bist ein Redakteur im Bereich Transport und Verkehr.
                    Du bis Experte dafür, relevante Schlagwörter für die Inhalte von Fachartikeln zu schreiben.
                    """
    task = f"""
            Erstelle maximal {max_keywords} Schlagworte.
            Die Antwort darf nur aus den eigentlichen Schlagworten bestehen.
            Das Format ist "Stichwort1, Stichwort2, Stichwort3, ..."
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
    cursor = collection.find({output_field: []})
    iteration = 0
    for record in cursor:
        iteration += 1
        if iteration > max_iterations:
            break
        article_text = record[input_field]
        if article_text == "":
            article_text = "Fehler: Kein Text vorhanden."
        else:
            embeddings = create_embeddings(text=article_text)
            collection.update_one({"_id": record['_id']}, {"$set": {output_field: embeddings}})
    print(f"\nGenerated embeddings for {iteration} records.")

def create_embeddings(text: str) -> list:
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).squeeze().tolist()

def ask_llm(llm: str, temperature: float = 0.2, question: str = "", history: list = [],
            systemPrompt: str = "", db_results_str: str = "", web_results_str: str = "") -> str:
    # define prompt
    datum_context = f" Heute ist der {str(datetime.now().date())}."
    input_messages = [
                {"role": "system", "content": systemPrompt + datum_context},
                {"role": "user", "content": question},
                {"role": "assistant", "content": 'Hier sind einige relevante Informationen aus dem DVV Artikel-Archiv:\n'  + db_results_str},
                {"role": "user", "content": "Gibt es zusätzliche Informationen aus dem Internet?"},
                {"role": "assistant", "content": 'Hier sind einige relevante Informationen aus einer Internet-Recherche:\n'  + web_results_str},
                {"role": "user", "content": 'Basierend auf den oben genannten Informationen, ' + question}
                ]
    if llm == "GPT 4o":
        response = openaiClient.chat.completions.create(
            model="gpt-4o",
            temperature=temperature,
            messages = input_messages
            )
        output = response.choices[0].message.content
    elif llm == "GPT 4o mini":
        response = openaiClient.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages = input_messages
            )
        output = response.choices[0].message.content
    elif llm == "anthropic":
        response = anthropicClient.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=systemPrompt,
            messages=input_messages[1:] # system prompt is not needed
        )
        output = response.content[0].text
    elif llm == "groq_mixtral-8x7b-32768":
        response = groqClient.chat.completions.create(
            model="mixtral-8x7b-32768",
            temperature=temperature,
            messages=input_messages
        )
        output = response.choices[0].message.content
    elif llm == "groq_llama3.1-70b":
        response = groqClient.chat.completions.create(
            model="llama-3.1-70b-versatile",
            temperature=temperature,
            messages=input_messages
        )
        output = response.choices[0].message.content
    elif llm == "groq_gemma-7b-it":
        response = groqClient.chat.completions.create(
            model="gemma-7b-it",
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
    return {field: {"$in": filter}} if filter else {}

def text_search(search_text : str = "*", score: float = 1.0, filter : list = [], limit : int = 10) -> tuple:
    if search_text == "":
        return []
    if search_text == "*":
        query = {
            "index": "volltext_gewichtet",
            # "sort": {"date": -1},
            "exists": {"path": "text"},
            }
    else:
        query = {
            "index": "volltext_gewichtet",
            # "sort": {"date": -1},
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
        "ki_abstract": 1,
        "score": {"$meta": "searchScore"},
        }
    pipeline = [
        {"$search": query},
        {"$match": {"quelle_id": {"$in": filter}}},
        # {"$match": {"score": {"$gte": score}}},
        {"$project": fields},
        {"$sort": {"date": -1}},
        {"$limit": limit},
        ]
    cursor = collection.aggregate(pipeline)
    return cursor

def vector_search(query_string: str = "*", score: float = 0.5, filter : list = [], sort: str = "date", limit: int = 10) -> tuple:
    embeddings_query = create_embeddings(text=query_string)
    query = {
            "index": "vector_index",
            "path": "text_embeddings",
            "queryVector": embeddings_query,
            "numCandidates": int(limit * 10),
            "limit": limit,
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
        {"$match": {"quelle_id": {"$in": filter}}},
        {"$sort": {sort: -1}},
        # {"$match": {"score": {"$gte": score}}},
        {"$project": fields},
        ]
    return collection.aggregate(pipeline)

def web_search_ddgs(query: str = "", limit: int = 10) -> list:
    # results = DDGS().text(f"Nachrichten über '{query}'", max_results=limit)
    results = DDGS().news(query, max_results=limit)
    return results if results else []

def web_search_tavily(query: str = "", score: float = 0.5, limit: int = 10) -> list:
    results: list = []
    results_list = tavilyClient.search(query=query, max_results=limit, include_raw_content=True)
    for result in results_list['results']:
        if result['score'] > score:
            results.append(result)
    return results

def print_results(cursor: list) -> None:
    if not cursor:
        print("Keine Artikel gefunden.")
    for item in cursor:
        print(f"[{str(item['datum'])[:10]}] {item['titel'][:70]}")

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

def get_systemprompt() -> str:
    result = collection_config.find_one({"key": "systemprompt"})
    return str(result.get("content"))
    
def update_systemprompt(text: str = ""):
    result = collection_config.update_one({"key": "systemprompt"}, {"$set": {"content": text}})
