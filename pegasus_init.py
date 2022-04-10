from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_dataset, list_datasets, load_metric, list_metrics
import torch

list_metrics()

Rouge = load_metric('rouge')
# Model and tokenizer
src_text = [
    """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
]

result = Rouge.compute(predictions=src_text, references=src_text)
print(result)
""" batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)


translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True) """

# Metric 