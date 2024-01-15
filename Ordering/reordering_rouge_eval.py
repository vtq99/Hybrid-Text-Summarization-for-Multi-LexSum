#!pip install rouge-score
from rouge_score import rouge_scorer

import json
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

sources_org_16k = []
targets = []
with open('/content/drive/MyDrive/multilexsum_data/validation_original_16k.jsonl', 'r') as f:
  for line in f:
    jline = json.loads(line)
    sources_org_16k.append(jline['source'])
    targets.append(jline['target'])

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for i, source in enumerate(sources_org_16k):
  r1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True).score(targets[i], source)['rouge1']
  r2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True).score(targets[i], source)['rouge2']
  rL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(targets[i], source)['rougeL']
  rouge1_scores.append([r1.precision, r1.recall, r1.fmeasure])
  rouge2_scores.append([r2.precision, r2.recall, r2.fmeasure])
  rougeL_scores.append([rL.precision, rL.recall, rL.fmeasure])

# Rouge score between truncated 16k source and target
print("     Precision    Recall    F-1")
print("R-1", np.mean(rouge1_scores, axis=0))
print("R-2", np.mean(rouge2_scores, axis=0))
print("R-L", np.mean(rougeL_scores, axis=0))

#--------------------------- Reorder MDS ---------------------------------
sources_MDS_16k = []
targets = []
with open('/content/drive/MyDrive/multilexsum_data/validation_MDS.jsonl', 'r') as f:
  for line in f:
    jline = json.loads(line)
    sources_MDS_16k.append(jline['source'])
    targets.append(jline['target'])

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for i, source in enumerate(sources_MDS_16k[:100]):
  r1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True).score(targets[i], source)['rouge1']
  r2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True).score(targets[i], source)['rouge2']
  rL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(targets[i], source)['rougeL']
  rouge1_scores.append([r1.precision, r1.recall, r1.fmeasure])
  rouge2_scores.append([r2.precision, r2.recall, r2.fmeasure])
  rougeL_scores.append([rL.precision, rL.recall, rL.fmeasure])
  
# Rouge score between 16k source (reordered MDS) and target
print("     Precision    Recall    F-1")
print("R-1", np.mean(rouge1_scores, axis=0))
print("R-2", np.mean(rouge2_scores, axis=0))
print("R-L", np.mean(rougeL_scores, axis=0))

#--------------------------- Reorder MDS+ ---------------------------------
sources_MDS_plus_16k = []
targets = []
with open('/content/drive/MyDrive/multilexsum_data/validation_MDS_plus.jsonl', 'r') as f:
  for line in f:
    jline = json.loads(line)
    sources_MDS_plus_16k.append(jline['source'])
    targets.append(jline['target'])

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for i, source in enumerate(sources_MDS_plus_16k):
  r1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True).score(targets[i], source)['rouge1']
  r2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True).score(targets[i], source)['rouge2']
  rL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(targets[i], source)['rougeL']
  rouge1_scores.append([r1.precision, r1.recall, r1.fmeasure])
  rouge2_scores.append([r2.precision, r2.recall, r2.fmeasure])
  rougeL_scores.append([rL.precision, rL.recall, rL.fmeasure])
  print(i, ' src done!')

# Rouge score between 16k source (reordered and redundancy removed across documents MDS+) and target
print("     Precision    Recall    F-1")
print("R-1", np.mean(rouge1_scores, axis=0))
print("R-2", np.mean(rouge2_scores, axis=0))
print("R-L", np.mean(rougeL_scores, axis=0))

#--------------------------- Reorder LDS ---------------------------------
sources_LDS_16k = []
targets = []
with open('/content/drive/MyDrive/multilexsum_data/validation_LDS.jsonl', 'r') as f:
  for line in f:
    jline = json.loads(line)
    sources_LDS_16k.append(' '.join(jline['source']))
    targets.append(jline['target'])

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for i, source in enumerate(sources_LDS_16k):
  r1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True).score(targets[i], source)['rouge1']
  r2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True).score(targets[i], source)['rouge2']
  rL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(targets[i], source)['rougeL']
  rouge1_scores.append([r1.precision, r1.recall, r1.fmeasure])
  rouge2_scores.append([r2.precision, r2.recall, r2.fmeasure])
  rougeL_scores.append([rL.precision, rL.recall, rL.fmeasure])

# Rouge score between 16k source (reordered and redundancy removed across documents LDS) and target
print("     Precision    Recall    F-1")
print("R-1", np.mean(rouge1_scores, axis=0))
print("R-2", np.mean(rouge2_scores, axis=0))
print("R-L", np.mean(rougeL_scores, axis=0))



"""\

\begin{equation}
   MMR_{Score} = \arg \max_{s_i\in D-S}[λ\ CosSim(s_i, Q) - (1-λ) \max_{s_j \in S} CosSim(s_i, s_j)] 
\end{equation}

\
"""

