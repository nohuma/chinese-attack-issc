# coding=utf-8
import argparse
import transformers

from utils.get_stopwords import get_stopwords
from utils.dataset import data_processor
from utils.zh_lstm_cnn import ZHLSTMForClassification, ZHWordCNNForClassification

import transformations
import search_methods

from textattack.models.wrappers import HuggingFaceModelWrapper, PyTorchModelWrapper
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification, InputColumnModification
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import TargetedClassification

from textattack import Attack, Attacker, AttackArgs
from textattack.datasets import Dataset, HuggingFaceDataset
import textattack


HUGGINGFACE_MODELS = {
    #
    # bert-base-chinese fine-tuned
    #
    "bert-chinanews": "models/bert-base-finetuned-chinanews-chinese",
    "bert-chnsenticorp": "models/bert-base-finetuned-chnsenticorp-chinese",
    "bert-ocnli": "models/bert-base-finetuned-ocnli-chinese",
    "bert-ctrip": "models/bert-base-finetuned-ctrip",
    "bert-jd": "models/bert-base-finetuned-jd",
    #
    # roberta-base-wwm-chinese-cluecorpussmall fine-tuned
    #
    "roberta-chinanews": "models/roberta-base-finetuned-chinanews-chinese",
    "roberta-chnsenticorp": "models/roberta-base-finetuned-chnsenticorp-chinese",
    "roberta-ocnli": "models/roberta-base-finetuned-ocnli-chinese",
    "roberta-ctrip": "models/roberta-base-finetuned-ctrip",
    "roberta-jd": "models/roberta-base-finetuned-jd",
    #
    # albert-base-chinese-cluecorpussmall fine-tuned
    #
    "albert-chinanews": "models/albert-base-finetuned-chinanews-chinese",
    "albert-chnsenticorp": "models/albert-base-finetuned-chnsenticorp-chinese",
    "albert-ocnli": "models/albert-base-finetuned-ocnli-chinese",
    "albert-lcqmc": "models/albert-base-finetuned-lcqmc-chinese",
    "albert-ctrip": "models/albert-base-finetuned-ctrip",
    "albert-jd": "models/albert-base-finetuned-jd",
    #
    # distilbert-base-multilingual-cased fine-tuned
    #
    "distilbert-chinanews": "models/distilbert-base-finetuned-chinanews-chinese",
    "distilbert-chnsenticorp": "models/distilbert-base-finetuned-chnsenticorp-chinese",
    "distilbert-ocnli": "models/distilbert-base-finetuned-ocnli-chinese",
    "distilbert-lcqmc": "models/distilbert-base-finetuned-lcqmc-chinese",
    "distilbert-ctrip": "models/distilbert-base-finetuned-ctrip",
    "distilbert-jd": "models/distilbert-base-finetuned-jd",
}
TEXTATTACK_MODELS = {
    #
    # LSTMs
    #
    "lstm-chinanews": "models/lstm_and_cnn/lstm-chinanews-chinese",
    "lstm-chnsenticorp": "models/lstm_and_cnn/lstm-chnsenticorp-chinese",
    #
    # CNNs
    #
    "cnn-chinanews": "models/lstm_and_cnn/cnn-chinanews-chinese",
    "cnn-chnsenticorp": "models/lstm_and_cnn/cnn-chnsenticorp-chinese",
}
DATA_PROCESSOR = {
    "chinanews": data_processor.ChinanewsProcessor,
    "chnsenticorp": data_processor.ChnsenticorpProcessor,
    "ocnli": data_processor.OcnliProcessor,
    "ctrip": data_processor.CtripHotelReviewsProcessor,
    "jd": data_processor.JDComProductReviewsProcessor,
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
}


def build_model_and_dataset(model_name_or_path):
    model_name, task_name = model_name_or_path.split('-')

    processor = DATA_PROCESSOR[task_name]()
    label_names = processor.get_labels()
    input_columns = processor.get_input_columns()

    dataset = processor.get_test_examples(
        f'attack_datasets/{task_name}')
    dataset = Dataset(dataset, input_columns=input_columns, label_names=label_names)

    if args.victim_model in HUGGINGFACE_MODELS.keys():
        model_name_or_path = HUGGINGFACE_MODELS[args.victim_model]

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path)
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    elif args.victim_model in TEXTATTACK_MODELS.keys():
        model_name_or_path = TEXTATTACK_MODELS[args.victim_model]

        if model_name == 'cnn':
            model = ZHWordCNNForClassification.from_pretrained(
                model_name_or_path)
            model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
        elif model_name == 'lstm':
            model = ZHLSTMForClassification.from_pretrained(model_name_or_path)
            model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    else:
        raise ValueError(f"{model_name_or_path} is not support!")

    return model_wrapper, dataset


def build_constraints(model_name_or_path):
    stopwords = get_stopwords()
    constraints = [RepeatModification(),
                   StopwordModification(stopwords=stopwords),]
                #    MaxWordsPerturbed(max_percent=0.15)]
    model_name, task_name = model_name_or_path.split('-')

    if task_name == 'ocnli':
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
    # elif task_name == 'lcqmc':
    #     input_column_modification = InputColumnModification(
    #         ["text_a", "text_b"], {"text_a"}
    #     )
    #     constraints.append(input_column_modification)
    else:
        pass

    return constraints


def main():
    # set random seed
    textattack.shared.utils.set_seed(718)
    model_wrapper, dataset = build_model_and_dataset(args.victim_model)
    # build attack
    constraints = build_constraints(args.victim_model)
    goal_function = TargetedClassification(model_wrapper, target_class=args.target_class, query_budget=50000)
    transformation = TRANSFORMATION_CLASS_NAMES[args.transformation]
    search_method = SEARCH_METHOD_CLASS_NAMES[args.search_method]
    attack = Attack(goal_function, constraints, transformation, search_method)
    
    attack_args = AttackArgs(
        num_examples=args.num_examples, random_seed=718, enable_advance_metrics=True, checkpoint_path=args.checkpoint_path)
    attacker = Attacker(attack, dataset, attack_args)
    # run attack
    attacker.attack_dataset()
    print(f"victim model: {args.victim_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformation",
        "-t",
        type=str,
        default='mix-ssc',
        choices=TRANSFORMATION_CLASS_NAMES.keys(),
        help="apply a transformation for attack"
    )
    parser.add_argument(
        "--search-method",
        "-s",
        type=str,
        default='ia',
        choices=SEARCH_METHOD_CLASS_NAMES.keys(),
        help="apply a search method for attack"
    )
    parser.add_argument(
        "--victim-model",
        "-m",
        type=str,
        default="bert-chinanews",
        help="apply a victim model and corresponding dataset for attack"
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=500,
        help="number of examples to attack"
    )
    parser.add_argument(
        "--checkpoint-path",
        "-ckp",
        type=str,
        default=None,
        help="path to load checkpoint files"
    )
    parser.add_argument(
        "--target-class",
        "-tgt",
        type=int,
        default=0,
        help="target label of dataset"
    )
    global args
    args = parser.parse_args()

    main()
