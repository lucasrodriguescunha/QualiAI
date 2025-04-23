from pymongo import MongoClient

# Conexão com o MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['quali_ai']
collection = db['resultados']