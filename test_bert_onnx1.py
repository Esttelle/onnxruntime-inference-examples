import logging
import os
import sys
import time
import torch
from itertools import chain
from argparse import Namespace

import evaluate
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
from onnxruntime.transformers import optimizer
from fusion_options import FusionOptions
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.export import Dim

from transformers import (
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers import (BertForSequenceClassification, BertTokenizer,)
from transformers.utils import check_min_version

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

def export_onnx(model, tokenizer, output_onnx_path):
    from transformers.onnx.features import FeaturesManager
    onnx_config = FeaturesManager._SUPPORTED_MODEL_TYPE['bert']['sequence-classification'](training_args)
    dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework='pt')
    torch.onnx.export(model,
        (dummy_inputs,),
        f=output_onnx_path,
        input_names=list(onnx_config.inputs.keys()),
        output_names=list(onnx_config.outputs.keys()),
        dynamic_axes={name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())},
        opset_version=17, 
        dynamo=False
    )
    input_names=list(onnx_config.inputs.keys())
    output_names=list(onnx_config.outputs.keys())
    # 定义动态维度
    batch_dim = Dim("batch", min=1, max=32)  # 批量维度，最小1，最大32
    seq_dim = Dim("seq", min=32, max=512)    # 序列维度，最小32，最大512

    # 定义 dynamic_shapes 字典
    dynamic_shapes = {
        'input_ids': {0: batch_dim, 1: seq_dim},
        'attention_mask': {0: batch_dim, 1: seq_dim},
        'token_type_ids': {0: batch_dim, 1: seq_dim}
    }
    # torch.onnx.export(model,
    #     (dummy_inputs,),
    #     f=output_onnx_path,
    #     input_names=input_names,
    #     output_names=output_names,
    #     opset_version=18, 
    #     dynamo=True
    # )

def preprocess_onnx(onnx_path, pre_onnx_path):
    # disable embedding layer norm optimization for better model size reduction
    opt_options = FusionOptions('bert')
    opt_options.enable_embed_layer_norm = False
    opt_options.intra_op_num_threads = 1
    opt_options.inter_op_num_threads = 1 

    quant_pre_process(onnx_path, 
                      pre_onnx_path, 
                      auto_merge=True,
                      optimization_options=opt_options)


def quantize_onnx(onnx_path, quant_onnx_path):
    quantize_dynamic(onnx_path, 
                     quant_onnx_path,  
                     extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT})
    

def evaluate_onnx(onnx_path, tokenizer):
    # load dataset
    raw_datasets = load_dataset("glue", configs.task_name, cache_dir=configs.cache_dir)
    sentence1_key, sentence2_key = task_to_keys[configs.task_name]

    def preprocess_function(examples):
        # Tokenize the texts
        args = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*args, padding='max_length', max_length=configs.max_seq_length, truncation=True)

        return result

    eval_dataset = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )

    # create onnx runtime session
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_path, sess_options,  providers=['CPUExecutionProvider'])

    # evaluation
    metric = evaluate.load("glue", configs.task_name, cache_dir=configs.cache_dir)

    def onnx_eval_step(batch):
        ort_inputs = {k: v.cpu().numpy() for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        ort_outs = ort_session.run(None, ort_inputs)
        return torch.tensor(ort_outs[0])

    dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=default_data_collator,
        shuffle=True
    )

    for batch in dataloader:
        logits = onnx_eval_step(batch)
        labels = batch['labels']
        metric.add_batch(predictions=logits.argmax(dim=-1), references=labels)

    eval_metric = metric.compute()
    print(f"ONNX model evaluation results: {eval_metric}")


def time_ort_model_evaluation(model_path, tokenizer):
    eval_start_time = time.time()
    evaluate_onnx(model_path, tokenizer)
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


def main():
    # get model
    tokenizer = BertTokenizer.from_pretrained(configs.output_dir)

    model = BertForSequenceClassification.from_pretrained(configs.output_dir)
    model.to(configs.device)

    # convert to onnx
    onnx_path = "bert_mrpc.onnx"
    export_onnx(model, tokenizer, onnx_path)

    # optimize model
    preprocessed_model_path = "bert_mrpc_optimized.onnx"
    preprocess_onnx(onnx_path, preprocessed_model_path)

    # quantize model
    quantized_model_path = "bert_mrpc_quant.onnx"
    quantize_onnx(preprocessed_model_path, quantized_model_path)

    print('ONNX full precision model size (MB):', os.path.getsize(onnx_path)/(1024*1024))
    time_ort_model_evaluation(onnx_path, tokenizer)

    print('ONNX quantized model size (MB):', os.path.getsize(quantized_model_path)/(1024*1024))
    time_ort_model_evaluation(quantized_model_path, tokenizer)


if __name__ == "__main__":
    main()