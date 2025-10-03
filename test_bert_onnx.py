from __future__ import absolute_import, division, print_function


import logging
import numpy as np
import os
import random
import sys
import time
import torch

from argparse import Namespace
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

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
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

#logging.getLogger("transformers.modeling_utils").setLevel(
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

# Set the device, batch size, topology, and caching flags.
configs.device = "cpu"
configs.eval_batch_size = 1
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False

# load model
model = BertForSequenceClassification.from_pretrained(configs.output_dir)
model.to(configs.device)

# define the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case)

# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

import onnxruntime

def export_onnx_model(args, model, tokenizer, onnx_model_path):
    with torch.no_grad():
        inputs = {'input_ids':      torch.ones(1,128, dtype=torch.int64),
                    'attention_mask': torch.ones(1,128, dtype=torch.int64),
                    'token_type_ids': torch.ones(1,128, dtype=torch.int64)}
        outputs = model(**inputs)

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,                                            # model being run
                    (inputs['input_ids'],                             # model input (or a tuple for multiple inputs)
                    inputs['attention_mask'], 
                    inputs['token_type_ids']),                                         # model input (or a tuple for multiple inputs)
                    onnx_model_path,                                # where to save the model (can be a file or file-like object)
                    opset_version=17,                                 # the ONNX version to export the model to
                    do_constant_folding=True,                         # whether to execute constant folding for optimization
                    input_names=['input_ids',                         # the model's input names
                                'input_mask', 
                                'segment_ids'],
                    output_names=['output'],                    # the model's output names
                    dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                'input_mask' : symbolic_names,
                                'segment_ids' : symbolic_names})
        logger.info("ONNX Model exported to {0}".format(onnx_model_path))

export_onnx_model(configs, model, tokenizer, "bert.onnx")

# optimize transformer-based models with onnxruntime-tools
import onnxruntime.transformers.optimizer as optimizer
from fusion_options import FusionOptions

# disable embedding layer norm optimization for better model size reduction
opt_options = FusionOptions('bert')
opt_options.enable_embed_layer_norm = False

opt_model = optimizer.optimize_model(
    'bert.onnx',
    'bert', 
    num_heads=12,
    hidden_size=768,
    optimization_options=opt_options)
opt_model.save_model_to_file('bert.opt.onnx')

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    from onnxruntime.quantization.shape_inference import quant_pre_process
    # quant_pre_process(onnx_model_path, onnx_model_path)
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8,
                     extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT})

    logger.info(f"quantized model saved to:{quantized_model_path}")

quantize_onnx_model('bert.opt.onnx', 'bert.opt.quant.onnx')

print('ONNX full precision model size (MB):', os.path.getsize("bert.opt.onnx")/(1024*1024))
print('ONNX quantized model size (MB):', os.path.getsize("bert.opt.quant.onnx")/(1024*1024))

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
    return dataset

def evaluate_onnx(args, model_path, tokenizer, prefix=""):

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(model_path, sess_options)

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        #eval_loss = 0.0
        #nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.detach().cpu().numpy() for t in batch)
            ort_inputs = {
                                'input_ids':  batch[0],
                                'input_mask': batch[1],
                                'segment_ids': batch[2]
                            }
            logits = np.reshape(session.run(None, ort_inputs), (-1,2))
            if preds is None:
                preds = logits
                #print(preds.shape)
                out_label_ids = batch[3]
            else:
                preds = np.append(preds, logits, axis=0)
                out_label_ids = np.append(out_label_ids, batch[3], axis=0)

        #print(preds.shap)
        #eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        #print(preds)
        #print(out_label_ids)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix + "_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def time_ort_model_evaluation(model_path, configs, tokenizer, prefix=""):
    eval_start_time = time.time()
    result = evaluate_onnx(configs, model_path, tokenizer, prefix=prefix)
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))

print('Evaluating ONNXRuntime full precision accuracy and performance:')
time_ort_model_evaluation('bert.opt.onnx', configs, tokenizer, "onnx.opt")
    
print('Evaluating ONNXRuntime quantization accuracy and performance:')
time_ort_model_evaluation('bert.opt.quant.onnx', configs, tokenizer, "onnx.opt.quant")