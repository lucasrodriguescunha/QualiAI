from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["quali_ai"]
colecao = db["resultados"]