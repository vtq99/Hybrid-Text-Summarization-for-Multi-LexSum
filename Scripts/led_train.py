import os
from datasets import load_dataset, load_metric
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM

from transformers import AutoTokenizer
from transformers import LEDModel, LEDForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import torch
import transformers

#torch.cuda.set_device(1)
transformers.logging.set_verbosity_info()

import warnings
warnings.filterwarnings('ignore')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_files = {"train": "data_full_formatted/train_full_formattated_16K.jsonl",
              "validation": "data_full_formatted/validation_full_formattated_16K.jsonl"}
dataset = load_dataset("json", data_files=data_files)

val = dataset["validation"]
train = dataset["train"]





def selected_labels(dataset):
    df = pd.DataFrame([], columns=["text", "summary"])

    formatted = Dataset.from_pandas(df)

    formatted = Dataset.from_pandas(df)
    for row in dataset:
        lines = row.get("text")
        indexes = row.get("label")
        res_list = [lines[i] for i in indexes]

        formatted = formatted.add_item({"text": res_list, "summary": row.get("summary")})

    return formatted

max_input_length = 16384
max_output_length = 1000
batch_size = 2

ckpt_led_base = "allenai/led-base-16384"
ckpt_led_large = "allenai/led-large-16384"

tokenizer = AutoTokenizer.from_pretrained(ckpt_led_base,   add_prefix_space=True)
model = LEDForConditionalGeneration.from_pretrained("attempt2/checkpoint-260")

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["source"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch




def selected_labels(dataset):
    df = pd.DataFrame([], columns=["source", "target"])

    formatted = Dataset.from_pandas(df)

    formatted = Dataset.from_pandas(df)
    for row in dataset:
        lines = row.get("source")
        indexes = row.get("label")
        res_list = [lines[i] for i in indexes]

        formatted = formatted.add_item({"text": res_list, "summary": row.get("summary")})

    return formatted



val_dataset = val.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=2
)

train_dataset = train.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=2
)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

led = AutoModelForSeq2SeqLM.from_pretrained("attempt2/checkpoint-260", gradient_checkpointing=True, use_cache=False)

# set generate hyperparameters
led.config.num_beams = 2
led.config.max_length = 1000
led.config.min_length = 100
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3

rouge = load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # Uncomment this line with GPU
    fp16=True,
    output_dir="attempt2",
    logging_steps=5,
    eval_steps=20,
    save_steps=20,
    save_total_limit=30,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
)

"""Just to test, we will do in a subset of the dataset."""

trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train("attempt2/checkpoint-260")

trainer.save_model()

