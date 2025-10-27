from __future__ import absolute_import, division, print_function


import logging
import numpy as np
import os
import random
import sys
import time
import torch
import json

import evaluate
from collections import Counter
from argparse import Namespace
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    default_data_collator,
    DataCollatorWithPadding
)
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
#from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
from datasets import load_dataset

# 检查量化是否成功的方法
def check_quantization_success(model):
    print(f"\n=== 量化状态检查 ===")
    
    quantized_layers = 0
    total_layers = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            total_layers += 1
            weight = module.weight
            
            print(f"\n{name}:")
            print(f"  权重数据类型: {weight.dtype}")
            print(f"  是否量化: {weight.is_quantized}")
            
            if weight.is_quantized:
                quantized_layers += 1
                print(f"  量化方案: {weight.qscheme()}")
                print(f"  缩放因子: {weight.q_scale()}")
                print(f"  零点: {weight.q_zero_point()}")
            else:
                print(f"  未量化 - 保持原始数据类型")
    
    print(f"\n=== 量化统计 ===")
    print(f"总层数: {total_layers}")
    print(f"量化层数: {quantized_layers}")
    print(f"量化比例: {quantized_layers/total_layers*100:.1f}%")

# Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Setup logging level to WARN. Change it accordingly
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARN)

# logging.getLogger("transformers.modeling_utils").setLevel(
#    logging.WARN)  # Reduce logging

print(torch.__version__)

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
configs.processor = processors[configs.task_name]()
configs.output_mode = output_modes[configs.task_name]
configs.label_list = configs.processor.get_labels()
configs.model_type = "bert".lower()
configs.do_lower_case = True

configs.cache_dir = os.path.join(configs.data_dir, 'cached_eval_{}_{}_{}'.format(
        list(filter(None, configs.model_name_or_path.split('/'))).pop(),
        str(configs.max_seq_length),
        str(configs.task_name)))
# Set the device, batch size, topology, and caching flags.
configs.device = "cpu"
configs.eval_batch_size = 1
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False
configs.max_eval_samples = None

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

sentence1_key, sentence2_key = task_to_keys[configs.task_name]

# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

# load model
model = BertForSequenceClassification.from_pretrained(configs.output_dir)
model.to(configs.device)

training_args = TrainingArguments(output_dir=configs.output_dir)

print(training_args)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/(1024*1024))
    os.remove('temp.p')


print_size_of_model(model)
# quantize model
# check_quantization_success(model)

quantize_(model, Int8DynamicActivationInt8WeightConfig())
quantized_model = torch.compile(model)

print_size_of_model(quantized_model)


def evaluate_model(args, model, tokenizer):
    raw_datasets = load_dataset(
            "glue",
            configs.task_name,
            cache_dir=configs.cache_dir
        )
    
    padding = "max_length"
    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        if "label" in examples:
            result["label"] = examples["label"]

        return result

    def debug_dataset(dataset):
        """
        调试数据集函数
        """
        print("=== 数据集调试信息 ===")
        print(f"数据集类型: {type(dataset)}")
        print(f"数据集分割: {list(dataset.keys())}")

        for split in dataset.keys():
            print(f"\n--- {split} 分割 ---")
            print(f"列名: {dataset[split].column_names}")
            print(f"样本数量: {len(dataset[split])}")

            if len(dataset[split]) > 0:
                first_item = dataset[split][0]
                print(f"第一条数据: {first_item}")
                print(f"第一条数据的键: {list(first_item.keys())}")

    # # 调试原始数据集
    # debug_dataset(raw_datasets)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        
    # print(raw_datasets)

    # # 调试tokenized数据集
    # debug_dataset(raw_datasets)

    eval_dataset = raw_datasets["validation_matched" if configs.task_name == "mnli" else "validation"]

    train_dataset = raw_datasets["train"]

    # # 手动验证tokenization是否工作
    # sample_data = train_dataset[:2]  # 取前2个样本
    # print("原始样本:", sample_data)

    # tokenized_sample = preprocess_function(sample_data)
    # print("Tokenized样本:", tokenized_sample)
    # print("Tokenized样本的键:", tokenized_sample.keys())

    # # 检查是否包含必要的字段
    # required_keys = ['input_ids', 'attention_mask']
    # for key in required_keys:
    #     if key in tokenized_sample:
    #         print(f"✓ 包含 {key}")
    #     else:
    #         print(f"✗ 缺少 {key}")

    # Labels
    if configs.task_name is not None:
        is_regression = configs.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name, cache_dir=args.cache_dir)
    elif is_regression:
        metric = evaluate.load("mse", cache_dir=args.cache_dir)
    else:
        metric = evaluate.load("accuracy", cache_dir=args.cache_dir)

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

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=tokenizer
    )
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [args.task_name]
    eval_datasets = [eval_dataset]
    print(eval_datasets)
    if args.task_name == "mnli":
        tasks.append("mnli-mm")
        valid_mm_dataset = raw_datasets["validation_mismatched"]
        if args.max_eval_samples is not None:
            max_eval_samples = min(len(valid_mm_dataset), args.max_eval_samples)
            valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
        eval_datasets.append(valid_mm_dataset)
        combined = {}

    for eval_data, task in zip(eval_datasets, tasks):
        # tokenize the dataset
        #eval_data = eval_data.map(tokenizer, batched=True)

        # eval_dataloader = trainer.get_eval_dataloader(eval_dataset=eval_data)
        # for step, inputs in enumerate(eval_dataloader):
        #     print(f"Step {step}: inputs keys = {list(inputs.keys())}")
        #     print(type(inputs))
        #     print(inputs)
        metrics = trainer.evaluate(eval_dataset=eval_data)

        max_eval_samples = (
            args.max_eval_samples if args.max_eval_samples is not None else len(eval_data)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_data))

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        print(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)


def evaluate_1(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    eval_datasets = [args.eval_dataset]
    for eval_task, eval_output_dir, eval_dataset in zip(eval_task_names, eval_outputs_dirs, eval_datasets):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        #eval_dataset = [args.eval_dataset]
        print("eval_task:")
        print(eval_task)
        print("eval_output_dir:")
        print(eval_output_dir)
        print("eval_dataset:")
        print(eval_dataset)
        # print("args.eval_dataset:")
        # print(TensorDataset(eval_data))

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        print(eval_sampler)
        print(eval_dataloader)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            #print(batch)
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    print(all_input_ids)
    print
    return dataset


def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    # result = evaluate(configs, model, tokenizer, prefix="")
    evaluate_model(configs, model, tokenizer)
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    # print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


# define the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case)

# Evaluate the original FP32 BERT model
# print('Evaluating PyTorch full precision accuracy and performance:')
# time_model_evaluation(model, configs, tokenizer)

# Evaluate the INT8 BERT model after the dynamic quantization
print('Evaluating PyTorch quantization accuracy and performance:')
time_model_evaluation(quantized_model, configs, tokenizer)

# Serialize the quantized model
quantized_output_dir = configs.output_dir + "quantized/"
if not os.path.exists(quantized_output_dir):
    os.makedirs(quantized_output_dir, exist_ok=True)
    tokenizer.save_pretrained(quantized_output_dir)
    quantized_model.save_pretrained(quantized_output_dir)
