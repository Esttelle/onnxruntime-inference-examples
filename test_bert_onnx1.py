import logging
import os
import sys
import torch
from itertools import chain
from collections import Counter
from argparse import Namespace

import evaluate
import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers import optimizer
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


def export_onnx(model, tokenizer, output_onnx_path):
    from transformers.onnx.features import FeaturesManager
    onnx_config = FeaturesManager._SUPPORTED_MODEL_TYPE['bert']['sequence-classification'](training_args)
    dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework='pt')
    torch.onnx.export(model,
        (dummy_inputs,),
        f=output_onnx_path,
        verbose=True,
        input_names=list(onnx_config.inputs.keys()),
        output_names=list(onnx_config.outputs.keys()),
        dynamic_axes={name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())},
        opset_version=17
    )


def main():
    # get model
    tokenizer = BertTokenizer.from_pretrained(configs.output_dir)

    model = BertForSequenceClassification.from_pretrained(configs.output_dir)
    model.to(configs.device)

    # convert to onnx
    onnx_path = "bert_mrpc.onnx"
    export_onnx(model, tokenizer, onnx_path)

    # optimize model
    optimized_model_path = "bert_mrpc_optimized.onnx"
    optimized_model = optimizer.optimize_model(
        onnx_path,
        model_type='bert',
        num_heads=12,
        hidden_size=768
    )
    optimized_model.save_model_to_file(optimized_model_path)
    # shape_inference.infer_shapes_path(onnx_path, inferred_model_path)
    # quantize onnx model
    quantized_model_path = "bert_mrpc_quant.onnx"
    quantize_dynamic(optimized_model_path, quantized_model_path, weight_type=QuantType.QInt8, extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT})

    # compare origin onnx model and quantized onnx model



if __name__ == "__main__":
    main()