from textattack.models.wrappers import PyTorchModelWrapper, HuggingFaceModelWrapper
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.datasets import Dataset
import textattack

from utils.dataset import data_processor
from textattack import Trainer, TrainingArgs, Attack, AttackArgs
from zh_lstm_cnn import ZHLSTMForClassification, ZHWordCNNForClassification

import transformers
import argparse
import os

from utils.get_stopwords import get_stopwords
import transformations 
import search_methods 

PROCESSOR = {
      "chinanews": data_processor.ChinanewsProcessor,
      "chnsenticorp": data_processor.ChnsenticorpProcessor,
      "ocnli": data_processor.OcnliProcessor,
      "ctrip": data_processor.CtripHotelReviewsProcessor,
      "jd": data_processor.JDComProductReviewsProcessor,
}

MODEL_PRETRAINED = {
    "bert":"google-bert/bert-base-chinese",
    "roberta":"uer/roberta-base-wwm-chinese-cluecorpussmall",
    "albert":"uer/albert-base-chinese-cluecorpussmall",
    "distilbert":"distilbert/distilbert-base-multilingual-cased"
}
TRANSFORMATION_CLASS_NAMES = {
    "shuffle": transformations.ChineseShuffleCharacterSwap(),
    "split": transformations.ChineseSplittingCharacterSwap(),
    "synonym": transformations.ChineseSynonymWordSwap(),
    "same-pinyin": transformations.ChineseHomophoneCharacterSwap(),
    "sim-pinyin": transformations.ChineseSimilarPinyinCharacterSwap(),
    "glyph": transformations.ChineseGlyphCharacterSwap(),
    "argot": transformations.ChineseArgotWordSwap,
    "wordnet": transformations.ChineseWordSwapWordNet('cmn'),
    "hownet": transformations.ChineseWordSwapHowNet(),
    "mix-ssc": transformations.mix_ssc,
    "mlm": transformations.ChineseWordSwapMaskedLM(), 
    "es": transformations.ChineseExpandingScopeWordSwap,
}
SEARCH_METHOD_CLASS_NAMES = {
    "greedy": textattack.search_methods.GreedySearch(),
    "ga": textattack.search_methods.AlzantotGeneticAlgorithm(),
    "delete": textattack.search_methods.GreedyWordSwapWIR(wir_method='delete'),
    "unk": textattack.search_methods.GreedyWordSwapWIR(wir_method='unk'),
    "pso": search_methods.FasterParticleSwarmOptimization(),
    'ia': search_methods.ImmuneAlgorithm(),
    'pwws': textattack.search_methods.GreedyWordSwapWIR(wir_method='weighted-saliency'),
}


def get_attack(attack_name, model_wrapper):
    attack_name = attack_name.lower()
    if attack_name == 'argot':
        transformation = TRANSFORMATION_CLASS_NAMES['argot']
        search_method = SEARCH_METHOD_CLASS_NAMES['delete']
    elif attack_name == 'ga':
        transformation = TRANSFORMATION_CLASS_NAMES['synonym']
        search_method = SEARCH_METHOD_CLASS_NAMES['ga']
    elif attack_name == 'pso':
        transformation = TRANSFORMATION_CLASS_NAMES['hownet']
        search_method = SEARCH_METHOD_CLASS_NAMES['pso']
    elif attack_name == 'beat':
        transformation = TRANSFORMATION_CLASS_NAMES['mlm']
        search_method = SEARCH_METHOD_CLASS_NAMES['unk']
    elif attack_name == 'es':
        transformation = TRANSFORMATION_CLASS_NAMES['es']
        search_method = SEARCH_METHOD_CLASS_NAMES['unk']
    elif attack_name == 'issc':
        transformation = TRANSFORMATION_CLASS_NAMES['mix-ssc']
        search_method = SEARCH_METHOD_CLASS_NAMES['ia']
    else:
        raise ValueError("Invalid attack method name!")
    
    goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper, query_budget=50000)
    stopwords = get_stopwords()
    constraints = [RepeatModification(), 
                   StopwordModification(stopwords=stopwords)]
    
    attack = Attack(goal_function, constraints, transformation, search_method)
    
    return attack
    

def main():
    processor = PROCESSOR[args.task_name]()
    path = os.path.join('attack_datasets', args.task_name)
    train_examples = processor.get_train_examples(path)
    dev_examples = processor.get_dev_examples(path)
    label_names = processor.get_labels()
    input_columns = processor.get_input_columns()

    train_examples = Dataset(train_examples, label_names=label_names, input_columns=input_columns)
    dev_examples = Dataset(dev_examples, label_names=label_names, input_columns=input_columns)

    if args.model_name_or_path in MODEL_PRETRAINED.keys():
        model_dir = MODEL_PRETRAINED[args.model_name_or_path]
    else:
        model_dir = args.model_name_or_path

    if args.model_name_or_path == "lstm":
        model = ZHLSTMForClassification(num_labels=len(label_names))
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    elif args.model_name_or_path == "cnn":
        model = ZHWordCNNForClassification(num_labels=len(label_names))
        model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    else:
        config = transformers.AutoConfig.from_pretrained(model_dir, num_labels=len(label_names), id2label={i: label for i, label in enumerate(label_names)},
        label2id = {label: i for i, label in enumerate(label_names)})
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, model_max_length=512)
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    
    # Adversarial training configuration
    if args.model_name_or_path == "lstm" or args.model_name_or_path == "cnn":
        training_args = TrainingArgs(
            num_epochs=10,
            learning_rate=2e-4,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            random_seed=718,
        )
    else:
        training_args = TrainingArgs(
            num_epochs=3,
            num_clean_epochs=1,
            learning_rate=3e-5,
            num_train_adv_examples=args.num_adv_examples,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            random_seed=718,
        )
    
    attack = get_attack(args.attack, model_wrapper)

    trainer = Trainer(
                model_wrapper,
                'classification',
                attack,
                train_examples,
                dev_examples,
                training_args,
            )
    trainer.train()
    print("{:=^50s}".format("Split Line"))
    print(f"{args.attack} method for adversarial training.")
    print(f"{args.num_adv_examples} examples for adversarial training.")
    print("{:=^50s}".format("Split Line"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Training')

    parser.add_argument('--task-name', '-t', default='jd', type=str,choices=PROCESSOR.keys(),
                        help='Dataset to finetune/train.')
    parser.add_argument('--model-name-or-path', '-model', default='bert', type=str, 
                        help='model name/path to finetune/train.')
    parser.add_argument('--num-adv-examples', '-nadv', default=500, type=int, 
                        help='number of adversarial examples to train.')
    parser.add_argument('--attack', '-a', default='argot', type=str, 
                        help='attack method for training.')
    
    global args
    args = parser.parse_args()

    main()


