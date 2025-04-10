# %%
from datasets import load_dataset

import softprompts as sp

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model, tokenizer = sp.get_model_and_tokenizer(model_name, device_map="cpu")

# %%
dataset = load_dataset("walledai/AdvBench", split="train")
messages = dataset["prompt"]
targets = dataset["target"]


# %%
softprompt, _ = sp.train_softprompt(
    model, tokenizer, messages, targets, sp.SoftPromptConfig(num_steps=10)
)
generation = sp.generate_with_softprompt(
    model, tokenizer, messages, softprompt, max_new_tokens=200
)

print(generation[0])
