import sys
import torch
import datasets
import csv
from utils import Cubert_Model, DEVICE, MAX_LEN
from utils import get_dataloader_input, get_dataloader
from utils import eval_fn, all_metrics_scores

# evaluation
span_model = Cubert_Model()
span_model.to(DEVICE)
span_model.load_state_dict(torch.load("finetuned_ckpts/Cubert-1K", map_location=DEVICE))

def get_EM(examples_data, model=span_model):
    assert len(examples_data) == 1
    (model_input_ids, model_segment_ids,
        model_input_mask, model_labels_ids) = get_dataloader_input(examples_data,
                                                                example_types_to_evaluate="all",
                                                                setting='ideal',
                                                                vocab_file="pretrained_model_configs/vocab.txt")
    target_sequences = model_labels_ids
    eval_data_loader, eval_file_length = get_dataloader(
        model_input_ids,
        model_input_mask,
        model_segment_ids,
        model_labels_ids
    )


    pruned_target_sequences, output_sequences, _ = eval_fn(
        eval_data_loader, model, DEVICE)

    pruned_target_sequences = pruned_target_sequences.tolist()
    output_sequences = output_sequences.tolist()

    metrics = all_metrics_scores(True, target_sequences,
                                    pruned_target_sequences, output_sequences)
    return metrics['exact_match']