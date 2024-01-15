import torch
import transformers
from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import AutoTokenizer
import json
import numpy as np

model = LEDForConditionalGeneration.from_pretrained("MahmoudAly/led-base-multi-lexsum-ordering").to("cuda")
tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384-multi_lexsum-source-long", truncation=True, truncation_side='right', model_max_length=16384)

contents = []
with open("test_LED_LDS.jsonl", mode='r', encoding='utf-8') as fr:
  for line in fr.readlines():
    j_lines = json.loads(line)
    contents.append(j_lines)

from torch.utils.checkpoint import checkpoint 
import warnings
warnings.filterwarnings('ignore')

for i, content in enumerate(contents):
  proxy = content["source"]
  inputs = tokenizer.encode(proxy, return_tensors="pt", truncation=True, max_length=16384).to("cuda")
  summary = model.generate(inputs, num_beams=5, max_length = 1024, no_repeat_ngram_size = 3)
  summary_decoded = tokenizer.decode(summary[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
  data = {'index': i,
          'target': content['target'],
          'prediction': summary_decoded}
  with open('test_LED_LDS_inference.jsonl', 'a') as f:
    f.write(json.dumps(data) + '\n')
    f.close()
  #print(i, 'src done')

#!pip install rouge-score
from rouge_score import rouge_scorer

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

with open('/content/drive/MyDrive/multilexsum_data/test_LED_inference.jsonl', 'r') as f:
  for line in f:
    jline = json.loads(line)
    prediction = jline['prediction']
    target = jline['target']
    r1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True).score(target, prediction)['rouge1']
    r2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True).score(target, prediction)['rouge2']
    rL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(target, prediction)['rougeL']
    rouge1_scores.append([r1.precision, r1.recall, r1.fmeasure])
    rouge2_scores.append([r2.precision, r2.recall, r2.fmeasure])
    rougeL_scores.append([rL.precision, rL.recall, rL.fmeasure])
    
print("     Precision    Recall    F-1")
print("R-1", np.mean(rouge1_scores, axis=0))
print("R-2", np.mean(rouge2_scores, axis=0))
print("R-L", np.mean(rougeL_scores, axis=0))