# %%
import os
import pickle
import json
from tqdm import tqdm
import datasets

# %%
query_data: dict = json.load(open("codequeries_meta.json", "r")) 
reccos_dict: dict = {q["name"]: q["reccomendation"] for q in query_data} 

dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TEST)
partitioned_data = {query['name']: dataset.filter(lambda x: x["query_name"] == query['name'] and x["example_type"] == 1) for query in query_data}

with open('partitioned_data_pos.pkl', 'wb') as f: 
    pickle.dump(partitioned_data, f)

# %%
def sample_data(all_data, query=None):
    df = all_data.to_pandas()
    # add filter for files with single example
    v = df.code_file_path.value_counts()
    df_filtered = df[df.code_file_path.isin(v.index[v.lt(2)])]
    all_data_filtered = datasets.Dataset.from_pandas(df_filtered)

    # # performance is good with examples with single answer span
    # examples_with_single_span = all_data_filtered.filter(lambda example: len(example["answer_spans"]) == 1)
    query_sample = (all_data_filtered.\
                                      shuffle(seed=42).\
                                      select(range(min(10, all_data_filtered.shape[0]))))
    # the following case doesn't occur
    if query_sample.shape[0] < 10:
        temp_data_to_add = all_data_filtered.filter(lambda example: len(example["answer_spans"]) > 1)
        assert query_sample.features.type == temp_data_to_add.features.type
        query_sample = datasets.concatenate_datasets([query_sample, temp_data_to_add])
    
    return query_sample

with open('partitioned_data_pos.pkl', 'rb') as f: 
  partitioned_data = pickle.load(f)
all_queries = list(partitioned_data.keys())

sampled_partitioned_data = {}
for query in all_queries:
   sampled_partitioned_data[query] = sample_data(partitioned_data[query], query)

with open('sampled_data_pos.pkl', 'wb') as f: 
    pickle.dump(sampled_partitioned_data, f)

#%%
with open('sampled_data_pos.pkl', 'rb') as f: 
  sampled_partitioned_data = pickle.load(f)
sampled_partitioned_data['Comparison of constants']['code_file_path']