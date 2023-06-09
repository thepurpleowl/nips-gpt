#%%
import pickle
from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path
import numpy as np
__FILE_DIR__ = Path(__file__).parent

def em_at_any(log_path, all_queries, query_folderName_map, n=10):
    cnt_total = 0
    total = 0
    _cols_ = ['i', 'query', 'file_path', 'prompt', 'ans_spans']
    _cols_.extend([f'ans_{i}' for i in range(n)])
    for query in all_queries:
        try:
            df = pd.read_csv(__FILE_DIR__ / f'{log_path}/{query_folderName_map[query]}_logs.csv',
                            names=_cols_, keep_default_na=False)
            assert df.shape[0] == 20 and df.shape[1] == len(_cols_)
        except (AssertionError, FileNotFoundError):
            # print(query)
            pass

        total += df.shape[0]
        cnt = 0
        for _, row in df.iterrows():
            actual_span = row['ans_spans']
            gen_spans = [row[x] for x in [f'ans_{i}' for i in range(n)] if row[x].strip()]
            found = False
            for span in gen_spans:
                if actual_span.strip() == span.strip():
                    cnt += 1
                    cnt_total += 1
                    found = True
                else:
                    if actual_span:
                        match = SequenceMatcher(None, actual_span, span).find_longest_match()
                        if (match.size/len(actual_span) > 0.9 and match.size/len(span) > 0.9):
                            cnt += 1
                            cnt_total += 1
                            found = True

                if found == True:
                    break
            # if not found:
            #     print(query, row['file_path'])
                # print(actual_span, gen_spans)
    print(f"Correct with any: {cnt_total}/{total} = {cnt_total/total}")

def cal_pass_at_k(n, c, k):
    if n -c <k:
        return 1
    return 1 - np.prod(1.0 - k/np.arange(n-c+1, n+1))

def pass_at_k(log_path, all_queries, query_folderName_map, k, n=10):
    pass_at_k = 0
    total = 0
    _cols_ = ['i', 'query', 'file_path', 'prompt', 'ans_spans']
    _cols_.extend([f'ans_{i}' for i in range(n)])
    for query in all_queries:
        try:
            df = pd.read_csv(__FILE_DIR__ / f'{log_path}/{query_folderName_map[query]}_logs.csv',
                            names=_cols_, keep_default_na=False)
            assert df.shape[0] == 20 and df.shape[1] == len(_cols_)
        except (AssertionError, FileNotFoundError):
            # print(query)
            pass
        total += df.shape[0]

        for _, row in df.iterrows():
            pass_cnt = 0
            actual_span = row['ans_spans']
            gen_spans = [row[x] for x in [f'ans_{i}' for i in range(n)] if row[x].strip()]
            for span in gen_spans:
                if actual_span.strip() == span.strip():
                    pass_cnt += 1
                else:
                    if actual_span:
                        match = SequenceMatcher(None, actual_span, span).find_longest_match()
                        if (match.size/len(actual_span) > 0.9 and match.size/len(span) > 0.9):
                            pass_cnt += 1
            assert pass_cnt <= 10

            # calculate pass@k
            pass_at_k += cal_pass_at_k(n, pass_cnt, k)
    print(f"Correct with pass@{k}: {pass_at_k}/{total} = {pass_at_k/total}")

def eval(log_path):
    n = 10
    with open(__FILE_DIR__ / 'resources/query_folderName_map.pkl', 'rb') as f:
        query_folderName_map = pickle.load(f)
    all_queries = list(query_folderName_map.keys())
    # all_queries = ['`__iter__` method returns a non-iterator']

    pass_at_k(log_path, all_queries, query_folderName_map, k=1)
    pass_at_k(log_path, all_queries, query_folderName_map, k=2)
    pass_at_k(log_path, all_queries, query_folderName_map, k=5)
    pass_at_k(log_path, all_queries, query_folderName_map, k=10)
    # em_at_any(log_path, all_queries, query_folderName_map)


if __name__ == "__main__":
    eval('test_dir_file_v3/logs')