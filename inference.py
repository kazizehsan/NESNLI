import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

print(torch.cuda.is_available())

# nesnli_small_trainer = "./nesnli-small-trainer/checkpoint-20"
# nesnli_trainer = "./nesnli-trainer/checkpoint-11660"
original_t5_small = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(original_t5_small)
model = T5ForConditionalGeneration.from_pretrained(original_t5_small)

raw_dataset = load_dataset("esnli")
raw_dataset = raw_dataset["test"].shuffle(seed=42).select(range(2))

def process_inputs_and_labels(example):
    # followed the MNLI input processing example in original T5 paper: https://arxiv.org/pdf/1910.10683.pdf
    example["processed_input"] = "nesnli hypothesis: " + example["hypothesis"] + " premise: " + example["premise"]
    
    # formatting the explanation to inspire a chain-of-thought like narrative
    example["processed_label"] = "A premise holds true that: \"" + example["premise"] + "\"." + \
    " Can that premise be followed by the hypothesis: \"" + example["hypothesis"] + "\"?" + \
    " An analysis of the premise and the hypothesis reveals the following. " + example["explanation_1"]
    
    return example

raw_dataset = raw_dataset.map(process_inputs_and_labels)

print(raw_dataset['processed_label'])
print()

sentences = [ x for x in raw_dataset['processed_input'] ]

inputs = tokenizer(sentences, return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
    max_new_tokens=512,
)

print()
print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))