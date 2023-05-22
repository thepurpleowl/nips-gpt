# %%
import os
import pickle
import json
from tqdm import tqdm
import datasets
import tiktoken
from cubert_spanprediction import get_EM

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

query_data: dict = json.load(open("resources/codequeries_meta.json", "r")) 
reccos_dict: dict = {q["name"]: q["reccomendation"] for q in query_data} 
dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TEST)
dataset.shape
# %%
dataset = datasets.load_dataset("thepurpleowl/codequeries", "ideal", split=datasets.Split.TEST)
# partitioned_data = {query['name']: dataset.filter(lambda x: x["query_name"] == query['name'] and x["example_type"] == 1) for query in query_data}
# with open('partitioned_data_pos.pkl', 'wb') as f: 
#     pickle.dump(partitioned_data, f)

partitioned_data_all = {query['name']: dataset.map(lambda x: count_file_tokens(x['code_file_path'])).filter(lambda x: x["query_name"] == query['name'] and x["file_tokens"] <3000 ) for query in query_data}

with open('resources/partitioned_data_all.pkl', 'wb') as f: 
    pickle.dump(partitioned_data_all, f)

#%%
with open('resources/partitioned_data_all.pkl', 'rb') as f: 
  partitioned_data_all = pickle.load(f)
all_queries = partitioned_data_all.keys()
total = 0
for query in all_queries:
   print(query, partitioned_data_all[query].shape[0])
   total += partitioned_data_all[query].shape[0]
total

#%%
sampling_metadata = {}
for query in tqdm(all_queries):
  sampling_metadata[query] = {}
  for row in partitioned_data_all[query]:
    em = int(get_EM([row]))
    if row['code_file_path'] in sampling_metadata[query]:
      assert row['example_type'] == sampling_metadata[query][row['code_file_path']]['example_type']
      if(sampling_metadata[query][row['code_file_path']]['em'] == em
          and em == 1):
        pass
      else:
         sampling_metadata[query][row['code_file_path']]['em'] = 0
    # create new entry if not exists          
    else:
       sampling_metadata[query][row['code_file_path']] = {
          'example_type': row['example_type'],
          'em': em,
          'file_tokens': count_file_tokens(row['code_file_path'])['file_tokens']
       }

with open('resources/sampling_metadata.pkl', 'wb') as f: 
    pickle.dump(sampling_metadata, f)

#%%

with open('resources/sampling_metadata.pkl', 'rb') as f: 
  sampling_metadata = pickle.load(f)
suc_total = 0
total = 0
querywise_files = {}
for query in sampling_metadata:
  querywise_files[query] = {}
  suc_pos = 0
  unsuc_pos = 0
  suc_neg = 0
  unsuc_neg = 0
  suc_pos_files = []
  unsuc_pos_files = []
  suc_neg_files = []
  unsuc_neg_files = []
  for ff in sampling_metadata[query]:
    if sampling_metadata[query][ff]['file_tokens'] < 2000:
      total += 1 
      if sampling_metadata[query][ff]['em'] == 1:
          suc_total += 1
          if sampling_metadata[query][ff]['example_type'] == 1:
            suc_pos += 1
            suc_pos_files.append(ff)
          else:
            suc_neg += 1
            suc_neg_files.append(ff)
      elif sampling_metadata[query][ff]['em'] == 0:
          if sampling_metadata[query][ff]['example_type'] == 1:
            unsuc_pos += 1
            unsuc_pos_files.append(ff)
          else:
            unsuc_neg += 1
            unsuc_neg_files.append(ff)
  
  querywise_files[query]['suc_pos_files'] = suc_pos_files
  querywise_files[query]['unsuc_pos_files'] = unsuc_pos_files
  querywise_files[query]['suc_neg_files'] = suc_neg_files
  querywise_files[query]['unsuc_neg_files'] = unsuc_neg_files
  print(query, suc_pos+suc_neg+unsuc_pos+unsuc_neg, suc_pos, suc_neg, unsuc_pos, unsuc_neg)      
print(f'{suc_total}/{total}', suc_total/total)


#%%
def get_from_other_bucket(all_files, bucket, bi):
   sample_bucket_order = list(range(bucket, 4)) + list(range(0, bucket))  
   assert len(sample_bucket_order) == 4
   for i in sample_bucket_order:
      if i != bucket:
         if bi[i] < len(all_files[i]):
            file_to_sample = all_files[i][bi[i]]
            bucket_of_sample = i
            return file_to_sample, bucket_of_sample
   return None, None

sampled_querywise_files = {}
for query in querywise_files:
  suc_pos_files =querywise_files[query]['suc_pos_files']
  unsuc_pos_files =querywise_files[query]['unsuc_pos_files']
  suc_neg_files =querywise_files[query]['suc_neg_files']
  unsuc_neg_files =querywise_files[query]['unsuc_neg_files']
  all_files = [suc_pos_files, unsuc_pos_files, suc_neg_files, unsuc_neg_files]
  total_all_fils = sum([len(all_files[i]) for i in range(4)])
  try:
    # print(query, [len(all_files[i]) for i in range(4)])
    assert total_all_fils > 20
  except AssertionError:
    print(query, sum([len(all_files[i]) for i in range(4)]))

  
  sampled_all_files =[[], [], [], []]
  bi = [0, 0, 0, 0]
  total_sampled = 0
  bucket = 0
  while(total_sampled < min(20, total_all_fils)):
    # bucket = total_sampled%4
    bucket = bucket %4
    if bi[bucket] < len(all_files[bucket]):
        sampled_all_files[bucket].append(all_files[bucket][bi[bucket]])
        bi[bucket] += 1
        bucket += 1
        total_sampled += 1
    else:
        o_file, o_bucket = get_from_other_bucket(all_files, bucket, bi)
        if o_bucket and o_file:
           sampled_all_files[o_bucket].append(o_file)
           assert o_file == all_files[o_bucket][bi[o_bucket]]
           bucket = o_bucket + 1
           bi[o_bucket] += 1
           total_sampled += 1
        else:
          bucket += 1
          pass
  sampled_querywise_files[query] = sampled_all_files
     
print('-'*50)
total = 0
suc_total = 0
for q in sampled_querywise_files:
   total += sum([len(sampled_querywise_files[q][i]) for i in range(4)])
   suc_total += len(sampled_querywise_files[q][0]) + len(sampled_querywise_files[q][2])
   if sum([len(sampled_querywise_files[q][i]) for i in range(4)]) < 20:
    print(q, 
          sum([len(sampled_querywise_files[q][i]) for i in range(4)]), 
          [len(sampled_querywise_files[q][i]) for i in range(4)])

print(f'{suc_total}/{total}', suc_total/total)

#%%
kxs = ['suc_pos_files', 'unsuc_pos_files', 'suc_neg_files', 'unsuc_neg_files']
for query in sampling_metadata:
  try:
     assert sum([len(set(sampled_querywise_files[query][i])) for i in range(4)]) == 20
  except AssertionError:
     print(query, sum([len(set(sampled_querywise_files[query][i])) for i in range(4)]))


with open('resources/sampled_querywise_files.pkl', 'wb') as f: 
    pickle.dump(sampled_querywise_files, f)

# %%
# OLD SAMPLING
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