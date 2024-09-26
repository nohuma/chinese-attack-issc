# chinese-attack-issc

This is the official repository for EMNLP 2024 paper "Adaptive Immune-based Sound-Shape Code Substitutions for Adversarial Chinese Text Attacks" by Ao Wang, Xinghao Yang, Chen Li, Baodi Liu, and Weifeng Liu.

## Installation

### Code

You can download the packages needed as follow:

```
pip install -r requirement.txt
```

Pay attention! 

We make some changes on the original textattack package, you have to replace it with ours. The changes are listed:

1. Prevent multiple jieba segmentations in the Chinese attack flow and segment it only once.
2. Add the metric output for Chinese text, e.g., multi-lingual USE, and support the BERTScore.
3. Support the Chinese WordNet candidates.
4. Save the checkpoint to the `checkpoints` folder when attack is done.

### Models and Datasets

We fine-tune victim models of several datasets based on the pre-training models.

You can download the models and datasets in https://huggingface.co/WangA and put the them to `models` and `attack_datasets`    folders respectively.

## Attacks

### Untargeted Attacks

You can conduct untargeted attack experiments as follow:

```
python untargeted_attack.py -t mix-ssc -s ia -m bert-chinanews -n 500
```

The logger file will be store at `checkpoints` folder, you can get the summary of this attack by runing:

```
python untargeted_attack.py -ckp YOUR_CHECKPOINT
```

You can find more details about it by checking the `untargeted_attack.py`.

### Targeted Attacks

You can conduct targeted attack experiments on `Chinanews`  by specifying a target label, e.g., 0, as follow:

```
python untargeted_attack.py -t mix-ssc -s ia -m bert-chinanews -n 500 -tgt 0
```

The logger file will be store at `checkpoints` folder.

You can find more details about it by checking the `targeted_attack.py`.

## Adversarial Training

You can conduct adversarial training experiments as follow:

1. Download the dataset from https://huggingface.co/WangA/attack_datasets and put them in `./attack_datasets`
2. Run the demo

```python
python adv_train.py -a issc -t jd -model bert -nadv 500
```

We use a default training setting for the adversarial training experiments. You can learn more details and customize it easily.

## Transfer Attacks

You can conduct transfer attacks for PLMs via a checkpoint as follow:

```python
python transfer_attack.py -model TARGET_MODEL_FOLDER -source SOURCE_MODEL -ckp YOUR_CHECKPOINT -d DATASET
```





