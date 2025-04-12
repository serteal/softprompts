from datasets import load_dataset

import softprompts as sp

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model, tokenizer = sp.get_model_and_tokenizer(model_name, device_map="auto")

dataset = load_dataset("walledai/AdvBench", split="train")
messages: list[str] = dataset["prompt"]
targets: list[str] = dataset["target"]

config = sp.SoftPromptConfig(num_steps=1, batch_size=1, num_epochs=2)
softprompt, _ = sp.train_softprompt(model, tokenizer, messages, targets, config)
generation = sp.generate_with_softprompt(
    model,
    tokenizer,
    messages,
    softprompt,
    batch_size=config.batch_size,
    max_new_tokens=200,
)

print(f"{'=' * 100}")
print(messages[0])
print(f"{'=' * 100}")
print(generation[0])
