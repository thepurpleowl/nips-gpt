#%%
from pathlib import Path
import pickle 
from model_handler import OpenAIModelHandler
from prompt_handler import BugBasherSimplePromptConstructor
import json
from log import Log
import os
from tqdm import tqdm


def extract_codeql_recommendation(meta_data):
    query_data: dict = json.load(open(meta_data, "r", encoding="utf-8"))
    reccos_dict: dict = {q["name"]: q["reccomendation"] for q in query_data}
    desc_dict: dict = {q["name"]: q["desc"] for q in query_data}

    return query_data, reccos_dict, desc_dict

query_data, reccos_dict, desc_dict = extract_codeql_recommendation("codequeries_meta.json")

model_config = {
    "engine": "gpt-35-turbo",  # "gpt-4"
    "model": "gpt-35-turbo",  # "gpt-4"
    "max_tokens": 4000,
    "temperature": 0.75,
    "n": 5,
    "stop": ['```'],
}

experiment_config = {
    "Target_folder": "test_dir_gt1",
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
    Path("key_fast.txt").read_text().strip(),  # openai_api_key
    timeout = experiment_config["timeout"],
    prompt_batch_size = experiment_config["prompt_batch_size"],
    max_attempts=experiment_config["max_attempts"],
    timeout_mf = experiment_config["timeout_mf"],
    openai_api_type="azure",
    openai_api_base="https://gcrgpt4aoai6.openai.azure.com/",
    openai_api_version="2023-03-15-preview",
    system_message=experiment_config["system_message"]
)

prompt_constructor = BugBasherSimplePromptConstructor(
    template_path="templateABC.j2",
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

def get_prompt_input(row):
    assert row['example_type'] == 1
    y = {}
    y['query_name'] = row['query_name']
    y['description'] = desc_dict[row['query_name']]
    y['recommendation'] = reccos_dict[row['query_name']]
    y['input_code'] = get_sanitized_content(row['context_blocks'])
    y['answer_spans'] = '\n'.join([ans['span'] for ans in row['answer_spans']])
    y['supporting_fact_spans'] = '\n'.join([sf['span'] for sf in row['supporting_fact_spans']])
    return y

with open('sampled_data_pos.pkl', 'rb') as f: 
  sampled_partitioned_data = pickle.load(f)
all_queries = list(sampled_partitioned_data.keys())

with open('query_folderName_map.pkl', 'rb') as f:
    query_folderName_map = pickle.load(f)

Logger = Log()
for query in tqdm(all_queries):
    query_folderName = query_folderName_map[query]
    try:
        os.makedirs(experiment_config["Target_folder"]+f"/logs")
    except FileExistsError:
        pass

    query_sample = sampled_partitioned_data[query]

    prompts, exp_results = [], []
    processed_rows = []
    i = 0
    for row in query_sample:
        prompt_input = get_prompt_input(row)
        prompt_str = prompt_constructor.construct(**prompt_input)
        if not prompt_str:
            continue

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
    