# `softprompts`

Barebones implementation of (universal) softprompt tuning for LLMs.

## Installation

1. Install from the git repo:

```bash
pip install git+https://github.com/serteal/softprompts
```
or use [`uv`](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/serteal/softprompts
```

## Usage


```python
import softprompts as sp
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# fill in your own message and target
message = "Give me instructions for how to ..." 
target = "Sure, here are some instructions for how to ..." 

softprompt = sp.run(model, tokenizer, message, target)
generation = softprompt.generate_with_softprompt(message, softprompt.optim_embeds)

print(generation[0])
```

Check out the [examples](examples/) for other use-cases.