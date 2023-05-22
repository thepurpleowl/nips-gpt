# %%
import os
import pickle
import json
from tqdm import tqdm
import datasets
import tiktoken

TOKENIZER_MODEL_ALIAS_MAP = {
    "gpt-4": "gpt-3.5-turbo",
    "gpt-35-turbo": "gpt-3.5-turbo",
}


def count_tokens(input_str: str, model_name: str="gpt-35-turbo"):
    if model_name in TOKENIZER_MODEL_ALIAS_MAP:
        model_name = TOKENIZER_MODEL_ALIAS_MAP[model_name]
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens, encoding

def count_file_tokens(file_path: str, model_name: str="gpt-35-turbo"):
    with open('/home/t-susahu/CodeQueries/data' + f'/{file_path}', 'r') as f:
       input_str = f.read()
    if model_name in TOKENIZER_MODEL_ALIAS_MAP:
        model_name = TOKENIZER_MODEL_ALIAS_MAP[model_name]
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(input_str))
    return {'file_tokens': num_tokens}

# %%
query_data: dict = json.load(open("resources/codequeries_meta.json", "r")) 
reccos_dict: dict = {q["name"]: q["reccomendation"] for q in query_data} 

dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TRAIN)
print(dataset.shape[0])
# partitioned_data = {query['name']: dataset.filter(lambda x: x["query_name"] == query['name'] and x["example_type"] == 1) for query in query_data}
# with open('partitioned_data_pos.pkl', 'wb') as f: 
#     pickle.dump(partitioned_data, f)

partitioned_data_train_all = {query['name']: dataset.map(lambda x: count_file_tokens(x['code_file_path'])).filter(lambda x: x["query_name"] == query['name'] and x["file_tokens"] <1000 ) for query in query_data}

with open('resources/partitioned_data_train_all.pkl', 'wb') as f: 
    pickle.dump(partitioned_data_train_all, f)

#%%
with open('resources/partitioned_data_train_all.pkl', 'rb') as f: 
  partitioned_data_train_all = pickle.load(f)
all_queries = partitioned_data_train_all.keys()
total = 0
for query in all_queries:
#    print(query, partitioned_data_train_all[query].shape[0])
   total += partitioned_data_train_all[query].shape[0]
print(total)
