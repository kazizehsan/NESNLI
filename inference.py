import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

print(torch.cuda.is_available())

nesnli_small_trainer = "./nesnli-small-trainer/checkpoint-20"

tokenizer = T5Tokenizer.from_pretrained(nesnli_small_trainer)
model = T5ForConditionalGeneration.from_pretrained(nesnli_small_trainer)

task_prefix = "nesnli "
# use different length sentences to test batching
sentences = [
    "hypothesis: A person is at a diner, ordering an omelette. premise: A person on a horse jumps over a broken down airplane.",
    "hypothesis: A person is training his horse for a competition. premise: A person on a horse jumps over a broken down airplane." 
]

inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)

print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))