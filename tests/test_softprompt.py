import pytest
import torch
from transformers import AutoTokenizer

from softprompts.softprompt import (
    add_optim_token_str_at_end,
    set_tokenizer_optim_token,
    split_tokenized_messages_on_optim_str,
    tokenize,
)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")


def test_set_tokenizer_optim_token_default(tokenizer):
    # Test with default token
    set_tokenizer_optim_token(tokenizer)
    assert hasattr(tokenizer, "optim_token_id")
    assert tokenizer.optim_token_id is not None
    assert tokenizer.convert_ids_to_tokens(tokenizer.optim_token_id) == "<|optim_str|>"


def test_set_tokenizer_optim_token_custom(tokenizer):
    # Test with custom token
    custom_token = "<|custom|>"
    set_tokenizer_optim_token(tokenizer, custom_token)
    assert hasattr(tokenizer, "optim_token_id")
    assert tokenizer.optim_token_id is not None
    assert tokenizer.convert_ids_to_tokens(tokenizer.optim_token_id) == custom_token


def test_add_optim_token_str_at_end():
    # Test with single string
    messages = "Hello world"
    result = add_optim_token_str_at_end(messages)
    assert result == ["Hello world<|optim_str|>"]

    # Test with list of strings
    messages = ["Hello", "World"]
    result = add_optim_token_str_at_end(messages)
    assert result == ["Hello<|optim_str|>", "World<|optim_str|>"]

    # Test with custom token
    messages = ["Hello", "World"]
    result = add_optim_token_str_at_end(messages, "<|custom|>")
    assert result == ["Hello<|custom|>", "World<|custom|>"]


def test_tokenize_basic(tokenizer):
    text = "Hello world"
    input_ids, attn_mask = tokenize(tokenizer, text)

    expected_input_ids = torch.tensor([[19556, 905]])
    expected_attn_mask = torch.tensor([[1, 1]])
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attn_mask, torch.Tensor)
    assert torch.all(input_ids == expected_input_ids)
    assert torch.all(attn_mask == expected_attn_mask)


def test_tokenize_list(tokenizer):
    texts = ["Hello", "World"]
    input_ids, attn_mask = tokenize(tokenizer, texts)
    assert input_ids.shape[0] == 2  # batch size 2
    assert attn_mask.shape[0] == 2


def test_tokenize_different_lengths(tokenizer):
    texts = ["Hello", "!World", "This is a long sentence"]
    # Should raise a ValueError for no padding set
    with pytest.raises(ValueError):
        tokenize(tokenizer, texts)

    # Should work
    correct_inp_ids, correct_attn_mask = tokenize(
        tokenizer, texts, padding=True, padding_side="left"
    )
    expected_inp_ids = torch.tensor(
        [[2, 2, 2, 2, 19556], [2, 2, 2, 17, 11265], [1348, 314, 253, 986, 6330]]
    )
    expected_attn_mask = torch.tensor(
        [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [1, 1, 1, 1, 1]]
    )
    assert torch.all(correct_inp_ids == expected_inp_ids)
    assert torch.all(correct_attn_mask == expected_attn_mask)


def test_split_tokenized_messages_on_optim_str(tokenizer):
    # Set up the tokenizer with the optimization token
    set_tokenizer_optim_token(tokenizer)

    messages = ["Hello", "This is a long sentence"]
    messages = add_optim_token_str_at_end(messages)
    input_ids, attn_mask = tokenize(
        tokenizer, messages, padding=True, padding_side="left"
    )

    # Split the messages
    left_ids, right_ids, left_mask, right_mask = split_tokenized_messages_on_optim_str(
        input_ids, attn_mask, tokenizer.optim_token_id
    )

    assert left_ids.shape[0] == 2
    assert right_ids.shape[0] == 2
    assert left_mask.shape[0] == 2
    assert right_mask.shape[0] == 2

    # Verify that optimization token is not in either side
    assert not torch.any(left_ids == tokenizer.optim_token_id)
    assert not torch.any(right_ids == tokenizer.optim_token_id)

    # Verify that the concatenation of left and right gives us back the original
    # (minus the optimization token)
    combined_ids = torch.cat([left_ids, right_ids], dim=1)
    original_without_optim = input_ids[input_ids != tokenizer.optim_token_id].reshape(
        2, -1
    )
    assert torch.all(combined_ids == original_without_optim)


def test_softprompt_single_iteration(tokenizer):
    from transformers import AutoModelForCausalLM

    from softprompts.softprompt import (
        SoftPrompt,
        SoftPromptConfig,
        generate_with_softprompt,
    )

    # Initialize model and config
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    config = SoftPromptConfig(
        num_steps=1,
        num_epochs=1,
        batch_size=2,  # two messages, one single iteration
        optim_str_init="x",
        seed=42,
        verbose=False,
    )

    softprompt = SoftPrompt(model, tokenizer, config)

    messages = ["Hello, how are you?", "What is your name?"]
    targets = ["I'm doing well, thank you!", "My name is Assistant."]

    optim_embeds, epoch_losses = softprompt.run(messages, targets)

    assert isinstance(optim_embeds, torch.Tensor)
    assert not optim_embeds.requires_grad  # Should be detached
    assert len(epoch_losses) == 1
    assert len(epoch_losses[0]) == 1
    assert len(epoch_losses[0][0]) == 1

    # The embedding dimension should match the model's embedding dimension
    assert optim_embeds.shape[1] == model.config.hidden_size
    # The sequence length should match the number of tokens in optim_str_init
    assert optim_embeds.shape[0] == len(tokenizer.tokenize(config.optim_str_init))

    # The loss should be a float and not NaN
    loss = epoch_losses[0][0][0]
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    assert loss > 0

    # Test generate with softprompt
    generated = generate_with_softprompt(
        model, tokenizer, messages, optim_embeds, max_new_tokens=10
    )
    assert isinstance(generated, list)
    assert len(generated) == 2
    assert isinstance(generated[0], str)
    assert isinstance(generated[1], str)
    assert generated[0] != generated[1]
