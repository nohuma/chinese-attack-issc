import os
import csv
import random


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    def get_input_columns(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class ChnsenticorpProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "train.tsv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            label = int(line[0])

            examples.append((text_a, label))

        return examples

    def get_dev_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "dev.tsv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            label = int(line[0])

            examples.append((text_a, label))

        return examples

    def get_test_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "test.tsv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            label = int(line[0])

            examples.append((text_a, label))

        return examples

    def get_labels(self):
        return ["negative", "positive"]

    def get_input_columns(self):
        return ["text"]


class LcqmcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "train.tsv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[0]
            text_b = line[1]
            label = int(line[2])

            examples.append(((text_a, text_b), label))

        return examples

    def get_dev_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "dev.tsv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[0]
            text_b = line[1]
            label = int(line[2])

            examples.append(((text_a, text_b), label))

        return examples

    def get_test_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "test.tsv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[0]
            text_b = line[1]
            label = int(line[2])

            examples.append(((text_a, text_b), label))

        return examples

    def get_labels(self):
        return ["different", "similar"]

    def get_input_columns(self):
        return ["text_a", "text_b"]


class ChinanewsProcessor(DataProcessor):
    def __init__(self, num_train_examples=50000, num_dev_examples=10000, num_test_examples=10000):
        self.num_train_examples = num_train_examples
        self.num_dev_examples = num_dev_examples
        self.num_test_examples = num_test_examples

    def get_train_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "train.csv")
        lines = self._read_tsv(data_path)

        random_lines = random.choices(lines, k=self.num_train_examples)

        for (i, line) in enumerate(random_lines):
            text_a = line[1] + '。' + line[2]
            label = int(line[0]) - 1

            examples.append((text_a, label))

        return examples

    def get_dev_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "train.csv")
        lines = self._read_tsv(data_path)

        random_lines = random.choices(lines, k=self.num_dev_examples)
        for (i, line) in enumerate(random_lines):
            text_a = line[1] + '。' + line[2]
            label = int(line[0]) - 1

            examples.append((text_a, label))

        return examples

    def get_test_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "test.csv")
        lines = self._read_tsv(data_path)

        random_lines = random.choices(lines, k=self.num_test_examples)
        for (i, line) in enumerate(random_lines):
            text_a = line[1] + '。' + line[2]
            label = int(line[0]) - 1

            examples.append((text_a, label))

        return examples

    def get_labels(self):
        return ["Mainland China Politics", "HongKong Macau Politics", "International News", "Financial News", "Culture",
                "Entertainment", "Sports"]

    def get_input_columns(self):
        return ["text"]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class OcnliProcessor(DataProcessor):
    def __init__(self, train_doc='train_50k.tsv'):
        self.train_doc = train_doc

    def get_train_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, self.train_doc)
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[0]
            text_b = line[1]
            label = int(line[2])

            examples.append(((text_a, text_b), label))

        return examples

    def get_dev_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "dev.tsv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[0]
            text_b = line[1]
            label = int(line[2])

            examples.append(((text_a, text_b), label))

        return examples

    def get_test_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "dev.tsv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[0]
            text_b = line[1]
            label = int(line[2])

            examples.append(((text_a, text_b), label))

        return examples

    def get_labels(self):
        return ["entailment", "neutral", "contradiction"]

    def get_input_columns(self):
        return ["premise", "hypothesis"]


class CtripHotelReviewsProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "train.csv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            text_a = line[1].replace(" ", "")
            label = int(line[0])

            if label == -1:
                label += 1
            examples.append((text_a, label))

        return examples

    def get_dev_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "test.csv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            text_a = line[1].replace(" ", "")
            label = int(line[0])

            if label == -1:
                label += 1
            examples.append((text_a, label))

        return examples

    def get_test_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "test.csv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            text_a = line[1].replace(" ", "")
            label = int(line[0])

            if label == -1:
                label += 1
            examples.append((text_a, label))

        return examples

    def get_labels(self):
        return ["negative", "positive"]

    def get_input_columns(self):
        return ["text"]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class JDComProductReviewsProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "train.csv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            text_a = line[1].replace(" ", "")
            label = int(line[0].replace('\ufeff', ''))

            if label == -1:
                label += 1
            examples.append((text_a, label))

        return examples

    def get_dev_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "test.csv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            text_a = line[1].replace(" ", "")
            label = int(line[0])

            if label == -1:
                label += 1
            examples.append((text_a, label))

        return examples

    def get_test_examples(self, data_dir):
        examples = []
        data_path = os.path.join(data_dir, "test.csv")
        lines = self._read_tsv(data_path)
        for (i, line) in enumerate(lines):
            text_a = line[1].replace(" ", "")
            label = int(line[0])

            if label == -1:
                label += 1
            examples.append((text_a, label))

        return examples

    def get_labels(self):
        return ["negative", "positive"]

    def get_input_columns(self):
        return ["text"]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

