from pymongo import MongoClient


MONGO_URI = "mongodb+srv://dhinakarshanmugamofficial_db_user:lkEuNuXY6WqFaa8h@cluster0.qtkjfq9.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)

db = client["animal_atc"]
collection = db["results"]

def save_result(data):
    collection.insert_one(data)
    
    
# lkEuNuXY6WqFaa8h - db password