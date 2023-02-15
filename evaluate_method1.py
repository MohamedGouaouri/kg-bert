import pandas as pd
from do_predict import predict_kgbert_singlepair

EVAL_DATA_FILE = './data/linkedin/eval_method1/eval_data.tsv'

df = pd.read_csv(EVAL_DATA_FILE, delimiter='\t', names=['head', 's1', 's2', 's3', 's4'])
# save a copy of the original df for evaluation
df_original = df.copy()

# all possible combinations of head and s1, s2, s3, s4
list_combinations = []
for triple in df.to_dict('records'):
    combinations = []
    combinations.append([triple['head'], triple['s1'], 0])
    combinations.append([triple['head'], triple['s2'], 0])
    combinations.append([triple['head'], triple['s3'], 0])
    combinations.append([triple['head'], triple['s4'], 0])
    list_combinations.append(combinations)
print(list_combinations)

id = 0

for combinations in list_combinations:
    for combination in combinations:
        # predict the score
        combination[2] = predict_kgbert_singlepair(combination[0], combination[1], id)
        id = id + 1
        print(combination)
    # rank 
    combinations.sort(key=lambda x: x[2], reverse=True)

# evaluate the rankings compared to the original df
score = 0
for i in range(len(list_combinations)):
    for j in range(len(list_combinations[i])):
        if list_combinations[i][j][0] == df_original.iloc[i]['head'] and list_combinations[i][j][1] == df_original.iloc[i]['s' + str(j+1)]:
            score += 1

# divide the score by the total number of predictions
score /= len(list_combinations) * 4
print(score)


