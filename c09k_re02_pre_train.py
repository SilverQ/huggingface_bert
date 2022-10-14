# import os
# import json
# import re
# import unicodedata
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tokenizers import *
# from sklearn.model_selection import train_test_split
# import nltk
# from nltk.data import load
import pickle
from transformers import *
from datasets import *

try:
    with open('c09k_170k_pre_train.pkl', 'rb') as f:
        d = pickle.load(f)
    print('Load completed')

except:
    corpus = Dataset.from_text('data/c09k_170k_corpus.txt')
    d = corpus.train_test_split(test_size=0.075)

    def dataset_to_text(corpus, output_filename="data.txt"):
        """Utility function to save dataset text to disk,
        useful for using the texts to train the tokenizer
        (as the tokenizer accepts files)"""
        with open(output_filename, "w") as f:
            for t in corpus["text"]:
                print(t, file=f)
#     dataset_to_text(d["train"], "data/c09k_170k_pre_train.txt")
#     dataset_to_text(d["test"], "data/c09k_170k_pre_test.txt")
    with open('c09k_170k_pre_train.pkl', 'wb') as f:
        pickle.dump(d, f)
    print('Load corpus and split completed')

# d["train"], d["test"]
# for t in d["train"]["text"][:3]:
#     print(t)
#     print("="*50)

tokenizer_path = 'c09k_tokenizer_2'
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, vocab_size=8000,
                                              local_files_only=True, lowercase=False, strip_accents=False)

max_length = 128
truncate_longer_samples = True
pre_trained_model_path = 'c09k_170k_pretrained'


def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"], truncation=True, padding="max_length",
                     max_length=max_length, return_special_tokens_mask=True)


def encode_without_truncation(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)


if truncate_longer_samples:
    encode = encode_with_truncation
else:
    encode = encode_without_truncation

train_dataset = d['train'].map(encode, batched=True)
test_dataset = d['test'].map(encode, batched=True)

if truncate_longer_samples:
    # remove other columns and set input_ids and attention_mask as PyTorch tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    # remove other columns, and remain them as Python lists
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

vocab_size = tokenizer.vocab_size
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=pre_trained_model_path,          # output directory to where save model checkpoint
    evaluation_strategy='steps',    # 'steps': evaluate each `logging_steps`, 'epoch'  : each epoch
    # overwrite_output_dir=False,
    resume_from_checkpoint=True,
    num_train_epochs=500.,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=64, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=8,  # evaluation batch size
    logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=500,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
