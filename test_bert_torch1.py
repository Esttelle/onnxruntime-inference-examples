#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers==4.57.0",
#     "accelerate >= 0.12.0",
#     "datasets >= 1.8.0",
#     "sentencepiece != 0.1.92",
#     "scipy",
#     "scikit-learn",
#     "protobuf",
#     "torch >= 1.3",
#     "evaluate",
# ]
# ///

"""Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import sys
import torch
from collections import Counter
from argparse import Namespace

import evaluate
import numpy as np
from datasets import load_dataset

from transformers import (
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers import (BertForSequenceClassification, BertTokenizer,)
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

configs = Namespace()

# The output directory for the fine-tuned model, $OUT_DIR.
configs.output_dir = "./MRPC/"

# The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
configs.data_dir = "./glue_data/MRPC"

# The model name or path for the pre-trained model.
configs.model_name_or_path = "bert-base-uncased"
# The maximum length of an input sequence
configs.max_seq_length = 128

# Prepare GLUE task.
configs.task_name = "MRPC".lower()
configs.device = "cpu"

# Where do you want to store the pretrained models downloaded from huggingface.co
configs.cache_dir = os.path.join(configs.data_dir, 'cached_eval_{}_{}_{}'.format(
    list(filter(None, configs.model_name_or_path.split('/'))).pop(),
    str(configs.max_seq_length),
    str(configs.task_name)))

training_args = TrainingArguments(
    output_dir=configs.output_dir,
    do_eval=True, use_cpu=True)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed)


def evaluate_model(model, tokenizer):
    # Preprocessing the raw_datasets
    raw_datasets = load_dataset("glue", configs.task_name)

    # Labels
    is_regression = configs.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
        
    sentence1_key, sentence2_key = task_to_keys[configs.task_name]

    # Padding strategy
    padding = "max_length"

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    elif not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}

    if configs.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({configs.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(configs.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    def print_class_distribution(dataset, split_name):
        label_counts = Counter(dataset["label"])
        total = sum(label_counts.values())
        logger.info(f"Class distribution in {split_name} set:")
        for label, count in label_counts.items():
            logger.info(f"  Label {label}: {count} ({count / total:.2%})")

    if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation_matched" if configs.task_name == "mnli" else "validation"]
    print_class_distribution(eval_dataset, "validation")

    # Get the metric function
    metric = evaluate.load("glue", configs.task_name, cache_dir=configs.cache_dir)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        if not training_args.eval_do_concat_batches:
            preds = np.concatenate(preds, axis=0)
            labels = np.concatenate(p.label_ids, axis=0)
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    data_collator = default_data_collator

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Evaluation
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [configs.task_name]
    eval_datasets = [eval_dataset]
    if configs.task_name == "mnli":
        tasks.append("mnli-mm")
        valid_mm_dataset = raw_datasets["validation_mismatched"]
        eval_datasets.append(valid_mm_dataset)
        combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("eval", metrics)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/(1024*1024))
    os.remove('temp.p')


def main():
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # config = AutoConfig.from_pretrained(
    #     configs.model_name_or_path,
    #     num_labels=num_labels,
    #     finetuning_task=configs.task_name,
    #     cache_dir=configs.cache_dir,
    #     revision="main",
    #     token=None,
    #     trust_remote_code=False
    # )
    # print(f"Config: {config}")
    tokenizer = BertTokenizer.from_pretrained(configs.output_dir)

    model = BertForSequenceClassification.from_pretrained(configs.output_dir)
    model.to(configs.device)

    evaluate_model(model, tokenizer)

    print_size_of_model(model)

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    print_size_of_model(quantized_model)
    evaluate_model(quantized_model, tokenizer)


if __name__ == "__main__":
    main()
