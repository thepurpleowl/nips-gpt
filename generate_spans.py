#%%
from pathlib import Path
import pickle 
from utils_openai import OpenAIModelHandler, PromptConstructor, Log
import json
import os
import logging
from datetime import datetime
from tqdm import tqdm

def get_timestamp():
    now = datetime.now()
    return now.strftime("%m-%d-%Y-%H-%M-%S")

logging.basicConfig(filename=f'qlog_{get_timestamp()}.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger('logger1')


def extract_codeql_recommendation(meta_data):
    query_data: dict = json.load(open(meta_data, "r", encoding="utf-8"))
    reccos_dict: dict = {q["name"]: q["reccomendation"] for q in query_data}
    desc_dict: dict = {q["name"]: q["desc"] for q in query_data}

    return query_data, reccos_dict, desc_dict

query_data, reccos_dict, desc_dict = extract_codeql_recommendation("resources/codequeries_meta.json")

model_config = {
    "engine": "gpt-35-tunro",  # "gpt-4"
    "model": "gpt-35-tunro",  # "gpt-4"
    "max_tokens": 4000,
    "temperature": 0.8,
    "n": 10,
    "stop": ['```'],
}

experiment_config = {
    "Target_folder": "test_dir_file_v3",
    "encoding":'UTF-8',
    "timeout": 8,
    "prompt_batch_size": 1,  # prompt_batch_size * max_tokens allowed in model < RateLimit (62*4000<250000)
    "timeout_mf": 2,
    "max_attempts": 12,
    "include_related_lines": True,
    "max_prompt_size": 4000,
    "extra_buffer": 200,
    "system_message": """Assistant is an AI chatbot that helps developers perform code quality related tasks. In particular, it can help with modifying input code, as per the suggestions given by the developers. It should not modify the functionality of the input code in any way. It keeps the changes minimal to ensure that the warning or bug is fixed. It does not introduce any performance improvements, refactoring, rewrites of code, other than what is essential for the task.""",
}

model_handler = OpenAIModelHandler(
    model_config,
    Path("resources/key_fast.txt").read_text().strip(),  # openai_api_key
    timeout = experiment_config["timeout"],
    prompt_batch_size = experiment_config["prompt_batch_size"],
    max_attempts=experiment_config["max_attempts"],
    timeout_mf = experiment_config["timeout_mf"],
    openai_api_type="azure",
    openai_api_base="https://gcrgpt4aoai6c.openai.azure.com/",
    openai_api_version="2023-03-15-preview",
    system_message=experiment_config["system_message"]
)

prompt_constructor = PromptConstructor(
    template_path="span_highlight_V3.j2",
    model=model_config["model"],
)

#%%
def get_sanitized_content(ctxt_blocks):
    context_blocks = ""
    for ctxt_block in ctxt_blocks:
        newline_removed_content = ("\n".join(line 
                                                for line in ctxt_block['content'].split('\n')
                                                if line))
        context_blocks += newline_removed_content
        context_blocks += '\n'
    return context_blocks

def get_prompt_input(row, file_level=False):
    assert row['example_type'] == 1
    y = {}
    y['query_name'] = row['query_name']
    y['description'] = desc_dict[row['query_name']]
    y['recommendation'] = reccos_dict[row['query_name']]
    if not file_level:
        y['input_code'] = get_sanitized_content(row['context_blocks'])
    else:
        with open(f"/home/t-susahu/CodeQueries/data/{row['code_file_path']}", 'r') as f:
            y['input_code'] = f.read()
    y['answer_spans'] = ':::-:::'.join([ans['span'] for ans in row['answer_spans']])
    y['supporting_fact_spans'] = ':::-:::'.join([sf['span'] for sf in row['supporting_fact_spans']])
    return y

def get_examples_values(query, pos_ex, neg_ex):
    y = {}
    assert query == pos_ex['query_name'] == neg_ex['query_name']
    y['positive_context'] = get_sanitized_content(pos_ex['context_blocks'])
    y['positive_spans'] = [ans['span'] for ans in pos_ex['answer_spans']]
    y['negative_context'] = get_sanitized_content(neg_ex['context_blocks'])

    return y

def get_file_level_prompt_input(query, file_path, partitioned_data_all):
    df = partitioned_data_all[query].filter(lambda example: example["code_file_path"] == file_path and example['query_name'] == query)
    y = {}
    y['query_name'] = query
    y['description'] = desc_dict[query]
    with open(f"/home/t-susahu/CodeQueries/data/{file_path}", 'r') as f:
        y['input_code'] = f.read()
    y['answer_spans'] = ''
    y['supporting_fact_spans'] = ''
    for row in df:
        y['answer_spans'] += ':::-:::'.join([ans['span'] for ans in row['answer_spans']])
        y['supporting_fact_spans'] += ':::-:::'.join([sf['span'] for sf in row['supporting_fact_spans']])
    if y['answer_spans']:
        for row in df:
            assert row['example_type'] == 1
    else:
        for row in df:
            assert row['example_type'] == 0
        y['answer_spans'] = 'N/A'
    return y

def get_filename(fn):
    return '_'.join(fn.split('/'))

def run(file_level):
    with open('resources/sampled_data_pos.pkl', 'rb') as f: 
        sampled_partitioned_data = pickle.load(f)
    all_queries = list(sampled_partitioned_data.keys())

    with open('resources/query_folderName_map.pkl', 'rb') as f:
        query_folderName_map = pickle.load(f)

    Logger = Log()
    for query in tqdm(all_queries):
        query_folderName = query_folderName_map[query]
        if not Path(experiment_config["Target_folder"]+f"/logs").exists():
            os.makedirs(experiment_config["Target_folder"]+f"/logs")
        if not Path(experiment_config["Target_folder"]+f"/{query_folderName}").exists():
            os.makedirs(experiment_config["Target_folder"]+f"/{query_folderName}")

        query_sample = sampled_partitioned_data[query]

        prompts, exp_results = [], []
        processed_rows = []
        i = 0
        for row in query_sample:
            logger.info(row['code_file_path'])
            prompt_input = get_prompt_input(row, file_level)
            prompt_str = prompt_constructor.construct(**prompt_input)
            if not prompt_str:
                logger.info(row['code_file_path'], prompt_str)
                continue

            with open(Path(experiment_config["Target_folder"]+f"/{query_folderName}/{get_filename(row['code_file_path'])}.log"), 'w') as f:
                f.write(prompt_str)

            prompts.append(prompt_str)

            original_responses = model_handler.get_response(prompt_str)
            exp_results.append(original_responses)
            p_row = [i, 
                    query,
                    row['code_file_path'],
                    '', #prompt_str,
                    row['answer_spans'][0]['span']]
            p_row.extend([
                            original_responses[i].text.strip() if original_responses[i].text else '' for i in range(model_config['n'])
                        ])
            processed_rows.append(p_row)
            i += 1
            
        # Logger.create_logs(prompts, exp_results, model_config)
        Logger.create_logs(experiment_config["Target_folder"]+f"/logs/{query_folderName}_logs.csv",
                        processed_rows)

import itertools

with open('resources/partitioned_data_all.pkl', 'rb') as f: 
  partitioned_data_all = pickle.load(f)
with open('resources/partitioned_data_train_all.pkl', 'rb') as f: 
  partitioned_data_train_all = pickle.load(f)

def get_examples_for_prompt(partitioned_data_train_all, query):
    pos_ex = partitioned_data_train_all[query].filter(lambda x: x["example_type"] == 1).shuffle(seed=42).select([0])
    neg_ex = partitioned_data_train_all[query].filter(lambda x: x["example_type"] == 0).shuffle(seed=42).select([0])
    assert not neg_ex[0]['answer_spans']
    assert not neg_ex[0]['supporting_fact_spans']

    return pos_ex[0], neg_ex[0]

def run2():
    with open('resources/sampled_querywise_files.pkl', 'rb') as f: 
        sampled_querywise_files = pickle.load(f)

    with open('resources/query_folderName_map.pkl', 'rb') as f:
        query_folderName_map = pickle.load(f)

    Logger = Log()
    all_queries = list(sampled_querywise_files.keys())
    # all_queries = ['`__iter__` method returns a non-iterator',]
    for query in tqdm(all_queries):
        logger.info(f'here is the current query: {query}')
        query_folderName = query_folderName_map[query]
        if not Path(experiment_config["Target_folder"]+f"/logs").exists():
            os.makedirs(experiment_config["Target_folder"]+f"/logs")
        if not Path(experiment_config["Target_folder"]+f"/{query_folderName}").exists():
            os.makedirs(experiment_config["Target_folder"]+f"/{query_folderName}")

        sampled_files = list(itertools.chain.from_iterable(sampled_querywise_files[query]))
        try:
            assert len(set(sampled_files)) == 20
        except AssertionError:
            logger.info(query, len(set(sampled_files)))
        pos_ex, neg_ex = get_examples_for_prompt(partitioned_data_train_all, query)
        example_values = get_examples_values(query, pos_ex, neg_ex)

        prompts, exp_results = [], []
        processed_rows = []
        i = 0
        for file_path in sampled_files:
            prompt_input = get_file_level_prompt_input(query, file_path, partitioned_data_all)
            for k in example_values:
                prompt_input[k] = example_values[k]
            prompt_str = prompt_constructor.construct(**prompt_input)
            if not prompt_str:
                logger.info(file_path, prompt_str)
                continue

            with open(Path(experiment_config["Target_folder"]+f"/{query_folderName}/{get_filename(file_path)}.log"), 'w') as f:
                f.write(prompt_str)

            prompts.append(prompt_str)

            original_responses = model_handler.get_response(prompt_str)
            exp_results.append(original_responses)
            p_row = [i, 
                    query,
                    file_path,
                    '', #prompt_str,
                    prompt_input['answer_spans']]
            p_row.extend([
                            original_responses[i].text.strip() if original_responses[i].text else '' for i in range(model_config['n'])
                        ])
            processed_rows.append(p_row)
            i += 1
            
        # Logger.create_logs(prompts, exp_results, model_config)
        Logger.create_logs(experiment_config["Target_folder"]+f"/logs/{query_folderName}_logs.csv",
                        processed_rows)
        
if __name__ == "__main__":
    run2()