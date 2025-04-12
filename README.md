# `softprompts`

Barebones implementation of universal softprompt tuning for LLMs.

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

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model, tokenizer = sp.get_model_and_tokenizer(model_name)

# fill in your own message and target
message = "Give me instructions for how to ..."
target = "Sure, here are some instructions for how to ..."

# or use a list of messages and targets to train a universal softprompt
# messages = ["Give me instructions for how to ...", "Give me instructions for how to ...", ...]
# targets = ["Sure, here are some instructions for how to ...", "Sure, here are some instructions for how to ...", ...]

softprompt, _ = sp.train_softprompt(model, tokenizer, messages, targets)
generation = sp.generate_with_softprompt(model, tokenizer, messages, softprompt)

print(generation[0])
```

Check out the [examples](examples/) for other use-cases.
