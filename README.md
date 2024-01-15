# Legal-Data-Science-Lab

## Description:
This repository contains the models and scripts of the work of the **legal nlp praktikum** for **Text Summarization** topic for WS 22/23. 

### Extractive model
**MemSum** model that we have trained can be found in the MemSum directory along side its scripts.

### Ordering approach
Ordering models and scripts for training and evaluation can be found in Ordering directory

### Abstractive model
The 2 **LED** models can be found in noise-injected model folder and normal led model folder. Scripts for training **LED** and evaluating it is found in Scripts directory

### Data
Our proxy labels we generated along with the noise-injected ones, budgeted at 4k (to fit **PRIMERA**) and 16k (to fit **LED**) can be found in Data directory

### Scripts
All significant notebooks such as creating proxy labels, evaluating pipeline, creating reformatted labels, and injecting noise to proxy labels can be found in Scripts directory

## Hugging face
Abstractive LED 2 models trained can be found on Hugging Face, and used immediately from "hebaabdelrazek/led-multilexsum-noise" and "hebaabdelrazek/led-multilexsum-normal"

It can be used with:
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("hebaabdelrazek/led-multilexsum-normal")

model = AutoModelForSeq2SeqLM.from_pretrained("hebaabdelrazek/led-multilexsum-normal")
```
