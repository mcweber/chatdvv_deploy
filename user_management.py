# ---------------------------------------------------
# Version: 16.06.2024
# Author: M. Weber
# ---------------------------------------------------
# 09.06.2024 Bug fixes. Implemented Vorname and Nachname.
# 16.06.2024 check_user() returns user object or empty string.
# ---------------------------------------------------

from datetime import datetime
import os
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Define constants ----------------------------------
load_dotenv()

mongoClient = MongoClient(os.environ.get('MONGO_URI_DVV'))
database = mongoClient.user_pool
collection_user_pool = database.users

# Functions -----------------------------------------

def add_user(user_name, user_pw) -> bool:
    try:
        collection_user_pool.insert_one({
            'username': user_name,
            'user_password': user_pw,
            'created': datetime.now()
        })
        return True
    except DuplicateKeyError:
        return False
    

def check_user(user_name, user_pw) -> str:
    user = collection_user_pool.find_one({
        'username': user_name,
        'user_password': user_pw
    })
    return user if user else ""


def delete_user(user_name) -> bool:
    collection_user_pool.delete_one({'username': user_name})
    return True


def list_users() -> list:
    users = collection_user_pool.find()
    return users

