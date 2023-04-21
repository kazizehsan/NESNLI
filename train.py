import os
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
import evaluate
from torch.nn import functional as F

print(torch.cuda.is_available())

raw_dataset = load_dataset("esnli")

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def process_inputs_and_labels(example):
    # followed the MNLI input processing example in original T5 paper: https://arxiv.org/pdf/1910.10683.pdf
    example["processed_input"] = "nesnli hypothesis: " + example["hypothesis"] + " premise: " + example["premise"]
    
    # formatting the explanation to inspire a chain-of-thought like narrative
    example["processed_label"] = "A premise holds true that: \"" + example["premise"] + "\"." + \
    " Can that premise be followed by the hypothesis: \"" + example["hypothesis"] + "\"?" + \
    " An analysis of the premise and the hypothesis reveals the following. " + example["explanation_1"]
    
    return example

raw_dataset = raw_dataset.map(process_inputs_and_labels)

print(raw_dataset["train"][0:2])

def tokenize_function(example):
    return tokenizer(
        example["processed_input"],
        text_target=example["processed_label"],
        max_length=350, # token length of the longest label is 350
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

tokenized_datasets = raw_dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=['premise','hypothesis','label','explanation_1','explanation_2','explanation_3','processed_input','processed_label']
)
tokenized_datasets

samples = tokenized_datasets["train"][2990:3005]
print([len(x) for x in samples["labels"]])

small_eval_dataset = tokenized_datasets["validation"].shuffle().select(range(80))

def compute_metrics(eval_preds):
    metric = evaluate.load("bleu")
    logits, labels = eval_preds
    
    labels[labels == -100] = tokenizer.pad_token_id
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    logits = torch.tensor(logits[0])
    probs = F.softmax(logits, dim=-1)
    idx_next = [torch.multinomial(unit_example, num_samples=1).view(-1) for unit_example in probs]
    idx_next[idx_next == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(idx_next, skip_special_tokens=True)
    
    return metric.compute(predictions=predictions, references=labels)


model_output_dir = "nesnli-trainer"

training_args = TrainingArguments(
    model_output_dir,
    
    # evaluation will run on entire `eval_dataset` passed in Trainer
    eval_accumulation_steps=5, # Number of predictions steps before moving output tensors to CPU
    evaluation_strategy="steps",
    eval_steps=1060, # Number of update steps between two evaluations, for use with #evaluation_strategy="steps"
    logging_steps=1060, # logging happens at eval_steps if logging_steps < eval_steps
    
    #num_train_epochs=3,
    max_steps=34336, # if set there will roughly only 1 epoch
    # but if gradient_accumulation_steps is also set, training wont stop early even if dataset is exhausted,
    # it will complete all max_steps. in fact, in this case, the number of epochs will increase
    
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    
    # increase this and reduce batch size in case of memory shortage
    # eval, log and save still happens at respective xxx_steps, not at gradient_accumulation_steps * xxx_steps
    # only the `Epoch` count is an indication that batches are being accumulated
    gradient_accumulation_steps=4,
    
    save_steps=1060, # Number of update steps between two saves. Reduce if each step is taking too long.
)


model = T5ForConditionalGeneration.from_pretrained("t5-small")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    # train_dataset=small_train_dataset,
    #eval_dataset=tokenized_datasets["validation"],
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


if os.path.exists(model_output_dir) and len(os.listdir(model_output_dir)):
    trainer.train(resume_from_checkpoint=True) # used to continue training
else:
    trainer.train() # used when starting training