from sentence_transformers import SentenceTransformer, util


skgbert = SentenceTransformer('./skgbert.out')


query = ["tensorflow_software"]
targets = ["deep learning", "flask backend framework", "python", "tensorflow", 'software']


query_embedding = skgbert.encode(query)
targets_embeddings = skgbert.encode(targets)

# print(dir(util))
scores = util.cos_sim(query_embedding, targets_embeddings)
print(scores)