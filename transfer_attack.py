import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
import tqdm

import torch

from utils.dataset import data_processor
from utils.zh_lstm_cnn import ZHLSTMForClassification, ZHWordCNNForClassification
from textattack.models.wrappers import HuggingFaceModelWrapper, PyTorchModelWrapper
from torch.utils.data import DataLoader

import transformers

import argparse
import os
import collections

PROCESSOR = {
      "chinanews": data_processor.ChinanewsProcessor,
      "chnsenticorp": data_processor.ChnsenticorpProcessor,
      "ocnli": data_processor.OcnliProcessor,
      "ctrip": data_processor.CtripHotelReviewsProcessor,
      "jd": data_processor.JDComProductReviewsProcessor,
}

ALL_MODEL = {
    "chinanews": {
        "cnn": "models/lstm_and_cnn/cnn-chinanews-chinese",
        "lstm": "models/lstm_and_cnn/lstm-chinanews-chinese",
        "bert": "models/bert-base-finetuned-chinanews-chinese",
        "roberta": "models/roberta-base-finetuned-chinanews-chinese",
        "albert": "models/albert-base-finetuned-chinanews-chinese",
        "distilbert": "models/distilbert-base-finetuned-chinanews-chinese",
    },
    "chnsenticorp": {
        "cnn": "models/lstm_and_cnn/cnn-chnsenticorp-chinese",
        "lstm": "models/lstm_and_cnn/lstm-chnsenticorp-chinese",
        "bert": "models/bert-base-finetuned-chnsenticorp-chinese",
        "roberta": "models/roberta-base-finetuned-chnsenticorp-chinese",
        "albert": "models/albert-base-finetuned-chnsenticorp-chinese",
        "distilbert": "models/distilbert-base-finetuned-chnsenticorp-chinese",
    },
    "ocnli": {
        "bert": "models/bert-base-finetuned-ocnli-chinese",
        "roberta": "models/roberta-base-finetuned-ocnli-chinese",
        "albert": "models/albert-base-finetuned-ocnli-chinese",
        "distilbert": "models/distilbert-base-finetuned-ocnli-chinese",
    },
    
}

def get_test_dataloader(dataset, batch_size):
    """Returns the :obj:`torch.utils.data.DataLoader` for evaluation.

    Args:
        dataset (:class:`~textattack.datasets.Dataset`):
            Dataset to use for evaluation.
        batch_size (:obj:`int`):
            Batch size for evaluation.
    Returns:
        :obj:`torch.utils.data.DataLoader`
    """
    # Helper functions for collating data
    def collate_fn(data):
        input_texts = []
        targets = []
        for _input, label in data:
            input_texts.append(_input)
            targets.append(label)
        return input_texts, torch.tensor(targets)

    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return eval_dataloader

def get_test_examples(ckp_path):
    ckp = textattack.shared.AttackCheckpoint.load(ckp_path)
    results = ckp.attack_log_manager.results

    test_examples = []
    for result in results:
        if isinstance(result, FailedAttackResult):
            continue
        elif isinstance(result, SkippedAttackResult):
            continue
        else:
            text = result.perturbed_result.attacked_text.text
            label = result.original_result.ground_truth_output
            example = (text, label)
            test_examples.append(example)
    
    print(f"{len(test_examples)} test examples are included.")
    return test_examples


def main():
    source_model = args.source_model
    model_name = args.model_name_or_path
    label_names = PROCESSOR[args.dataset]().get_labels()
    model_dir = ALL_MODEL[args.dataset][args.model_name_or_path]

    test_examples = get_test_examples(args.checkpoint)
    
    
    if model_name == 'cnn':
        model = ZHWordCNNForClassification.from_pretrained(model_dir)
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
        dataloader = get_test_dataloader(test_examples, 16)
    elif model_name == 'lstm':
        model = ZHLSTMForClassification.from_pretrained(model_dir)
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
        dataloader = get_test_dataloader(test_examples, 16)
    else:
        config = transformers.AutoConfig.from_pretrained(model_dir, num_labels=len(label_names))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, model_max_length=512)
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        
        dataloader = DataLoader(test_examples, 16)

    model.to(device)
    model.eval()

    total_acc = 0

    prog_bar = tqdm.tqdm(
        dataloader,
        desc="Iteration",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    for _, batch in enumerate(prog_bar):
        inputs_list, targets = batch
        targets = targets.to(device)
        logits = model_wrapper(inputs_list)
        logits = torch.tensor(logits).to(device)
        labels = torch.argmax(logits, dim=1).squeeze()
        acc = torch.eq(labels, targets).float().sum().item()
        total_acc += acc
    
    assert logits.shape[-1] == len(label_names), f"label output {logits.shape[-1]} not equal to {len(label_names)}"
    print(f'Transfer attack from {source_model} to {args.model_name_or_path}')
    print('Test acc: {:.2f} %'.format(total_acc / len(test_examples) * 100))
    print(f"num of test examples are {len(test_examples)}")
    print(f"checkpoint: {args.checkpoint}")


if __name__ == '__main__':
    torch.manual_seed(718)
    torch.cuda.manual_seed(718)
    parser = argparse.ArgumentParser(description='Transfer attack')

    parser.add_argument('--model-name-or-path', '-model', default='roberta', type=str,
                        help='Target model to test')
    parser.add_argument('--checkpoint', '-ckp', required=True, type=str,
                        help='checkpoint including datasets.')
    parser.add_argument('--dataset', '-d', default='chinanews', type=str,
                        help='dataset to test')
    parser.add_argument('--source-model', '-source', default='bert', type=str,
                        help='Model that the adversarial examples generated')

    global args, device
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()
