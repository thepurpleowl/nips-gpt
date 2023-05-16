# %%
import csv
fields = ['Query', 'File', 'Prompt', 'Answer_span', 'Marked_span'] 


class Log():
    '''Class to log the results of the experiment in human readable format'''
    def __init__(self):
        pass
    
    # def create_logs(self, prompts, exp_results, model_config):
    #     with open(self.exp_log_lib + f"/{self.fname}.log", "w") as f:
    #         f.write(f"Results for experiment with parameters -\n{model_config}\n")
    #         for i, exp_result in enumerate(exp_results):
    #             f.write(f"\n\n\n======================== Prompt {i} ========================\n")
    #             f.write(prompts[i])
    #             f.write(f"\n------------ Result for Prompt {i} ------------\n\n")
    #             f.write(exp_result[0].text)
    #             f.write(f"\n\n------------ End Result for Prompt {i} ------------\n\n")
    #             f.write(f"Finish Reason: {exp_result[0].finish_reason}\n\n")
    #             f.write(f"======================== End Prompt {i} ========================\n")

    def create_logs(self, log_file_path, rows):
        with open(log_file_path, "w") as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
