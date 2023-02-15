import pandas as pd
from do_predict import predict_kgbert_singlepair
import time
start = time.time()

EVAL_DATA_FILE = './data/linkedin/eval_method2/__eval_data.tsv'
ENTITIES_FILE = './data/linkedin/entities.txt'
evaluation_score = 0

# get all eval pairs
df = pd.read_csv(EVAL_DATA_FILE, delimiter='\t', names=['head', 'tail'])
number_of_eval_pairs = len(df.index)
# get all entities from entities.txt
df_entities = pd.read_csv(ENTITIES_FILE, delimiter='\t', names=['entity'])

id=0
# group lines by head
df_grouped = df.groupby('head')
# iterate over groups
for head, group in df_grouped:
    
    scores = []
    # gather scores for head with each entity
    for entity in df_entities['entity'].tolist() :
        # print(head, entity)
        # predict the score
        score = predict_kgbert_singlepair(head, entity, id)
        id = id + 1
        # print(score)
        scores.append(score)
        
    scores.sort(reverse=True)

    # get the list of eval tails
    tails = group['tail'].tolist()
    for tail in tails : 
        print(head, tail)
        # predict the score
        score = predict_kgbert_singlepair(head, tail, id)
        id = id + 1
        # ensure that the score is in the list
        scores.append(score)
        scores.sort(reverse=True)
        # get rank of the score
        rank = scores.index(score)
        print(score, rank)
        # add 1 to the evaluation score if the rank is less than 100
        if rank < 100 :
            evaluation_score += 1
        

elapsed = time.time() - start
print("Elapsed time:", elapsed) 
print("Evaluation score:", evaluation_score)
print("Evaluation score normalized:", evaluation_score / (number_of_eval_pairs * 100))



