import logging
from dataclasses import dataclass

import torch
import transformers
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformers import set_seed

logger = logging.getLogger("softprompts")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class SoftPromptConfig:
    num_steps: int = 100
    num_epochs: int = 1
    batch_size: int = 4
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    lr: float = 1e-4
    seed: int | None = None
    verbose: bool = True


def set_tokenizer_optim_token(
    tokenizer: transformers.PreTrainedTokenizer,
    optim_token_str: str = "<|optim_str|>",
) -> None:
    """Sets the optimization token for a tokenizer."""
    tokenizer.add_special_tokens({"additional_special_tokens": [optim_token_str]})
    tokenizer.optim_token_id = tokenizer.convert_tokens_to_ids(optim_token_str)


def get_model_and_tokenizer(
    model_name_or_path: str,
    optim_token_str: str = "<|optim_str|>",
    device_map: str = "auto",
    **model_kwargs,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Get a model and tokenizer for a given model name or path.
    Freezes model parameters and sets padding options in the tokenizer.
    Also, creates a new token that we will use to insert the soft prompt in the messages.

    Args:
        model_name_or_path: The name or path of the model to load.
        optim_token_str: The string to use for the soft prompt.
            Should be left as default unless explicitly required.
        device_map: The device map to use for the model.
            Passed to transformers' `from_pretrained`.
        **model_kwargs: Additional keyword arguments to pass to the model.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map=device_map, **model_kwargs
    )
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    set_tokenizer_optim_token(tokenizer, optim_token_str)

    # Ensure the tokenizer has a pad token and set padding options
    tokenizer.padding_side = "left"
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def add_optim_token_str_at_end(
    messages: list[str], optim_token_str: str = "<|optim_str|>"
) -> list[str]:
    """Adds the <|optim_str|> token to each message."""
    return [x + optim_token_str for x in messages]


def tokenize(
    tokenizer: transformers.PreTrainedTokenizer,
    text: str | list[str],
    add_chat_template: bool = False,
    **tokenizer_kwargs,
) -> tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch seq_len"]]:
    """Wraps tokenizer's call by adding chat template if necessary and removing extra BOS tokens.
    If the text is a string, it is wrapped in a list.

    Args:
        tokenizer: The tokenizer to use.
        text: The text to tokenize.
        add_chat_template: Whether to add a chat template.
    """
    if isinstance(text, str):
        text = [text]

    if add_chat_template:
        # Assume that text argument is the user input
        batched_messages = [[{"role": "user", "content": msg}] for msg in text]
        text = tokenizer.apply_chat_template(
            batched_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if tokenizer.bos_token:
            # Remove the BOS token -- this will get added when tokenizing, if necessary
            text = [
                text.replace(tokenizer.bos_token, "")
                for text in text
                if text.startswith(tokenizer.bos_token)
            ]

    inputs = tokenizer(
        text=text,
        return_tensors="pt",
        **tokenizer_kwargs,
    )

    return inputs["input_ids"], inputs["attention_mask"]


def split_tokenized_messages_on_optim_str(
    input_ids: Int[Tensor, "batch seq_len"],
    attn_mask: Int[Tensor, "batch seq_len"],
    optim_token_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Splits the tokenized messages on the <|optim_str|> token.
    Returns the input ids and mask at left and right of the optim token.
    Input ids are assumed to be left-padded.
    Optim token is expected to be in the same column in all rows.

    Args:
        input_ids: The tokenized messages.
        attn_mask: The attention mask.
        optim_token_id: The id of the optim token.
            Can be retrieved as tokenizer.optim_token_id if using `get_model_and_tokenizer()`.
    """
    optim_str_positions = (input_ids == optim_token_id).nonzero(as_tuple=False)
    unique_columns = torch.unique(optim_str_positions[:, 1])
    counts = torch.sum(input_ids == optim_token_id, dim=1)
    assert torch.all(counts == 1), (
        "Not exactly one instance of optim_token_id in each row."
    )
    assert len(unique_columns) == 1, (
        "optim_token_id is not in the same column in all rows."
    )

    insertion_column = unique_columns.item()
    left_input_ids = input_ids[:, :insertion_column]
    right_input_ids = input_ids[:, insertion_column + 1 :]
    left_attn_mask = attn_mask[:, :insertion_column]
    right_attn_mask = attn_mask[:, insertion_column + 1 :]
    return left_input_ids, right_input_ids, left_attn_mask, right_attn_mask


def generate_with_softprompt(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: str | list[str],
    optim_embeds: Tensor,
    optim_token_str: str = "<|optim_str|>",
    batch_size: int = 8,
    **generation_kwargs,
) -> str | list[str]:
    """Wrapper around model.generate that injects the softprompt at the end of the message.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        messages: The messages to generate a response to.
        optim_embeds: The softprompt to use.
        optim_token_str: The string to use for the softprompt.
            Should be left as default unless explicitly required.
        **generation_kwargs: Additional keyword arguments to pass to the model.
    """
    if isinstance(messages, str):
        messages = [messages]

    if tokenizer.optim_token_id is None:
        logger.warning(
            "Tokenizer does not have an optimization token id. Adding it manually."
        )
        set_tokenizer_optim_token(tokenizer, optim_token_str)

    if model.device == torch.device("cpu"):
        logger.warning("Running on CPU -- this will be slow!")

    dataloader = torch.utils.data.DataLoader(
        messages, batch_size=batch_size, shuffle=False
    )

    generations_list = []
    for batched_messages in tqdm(dataloader, desc="Generating", total=len(dataloader)):
        # Add optimization token, pad & tokenize, and split the messages
        batched_messages = add_optim_token_str_at_end(batched_messages, optim_token_str)
        input_ids, attn_mask = tokenize(
            tokenizer,
            batched_messages,
            add_chat_template=True,
            padding="longest",
            padding_side="left",
        )
        input_ids = input_ids.to(model.device)
        attn_mask = attn_mask.to(model.device)
        left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = (
            split_tokenized_messages_on_optim_str(
                input_ids, attn_mask, tokenizer.optim_token_id
            )
        )

        before_embeds, after_embeds = [
            model.get_input_embeddings()(ids)
            for ids in (left_input_ids, right_input_ids)
        ]

        optim_embeds = optim_embeds.to(model.device)
        batched_optim_embeds = optim_embeds.expand(len(batched_messages), -1, -1)
        batched_attn_mask = torch.ones(
            len(batched_messages), optim_embeds.shape[1], device=model.device
        )

        input_embeds = torch.cat(
            [
                before_embeds.detach(),
                batched_optim_embeds,
                after_embeds.detach(),
            ],
            dim=1,
        )
        input_attn_mask = torch.cat(
            [
                left_attn_mask.detach(),
                batched_attn_mask,
                right_attn_mask.detach(),
            ],
            dim=1,
        )

        generation = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_attn_mask,
            **generation_kwargs,
        )
        generations_list.extend(generation)

    return tokenizer.batch_decode(generations_list, skip_special_tokens=False)


class SoftPrompt:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: SoftPromptConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        if tokenizer.optim_token_id is None:
            logger.warning(
                "Tokenizer does not have an optimization token id. Adding it manually."
            )
            set_tokenizer_optim_token(tokenizer)

        self.device = model.device
        if self.device == torch.device("cpu"):
            logger.warning("Running on CPU -- this will be slow!")

        if self.config.seed is not None:
            set_seed(self.config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

    def run(
        self,
        messages: str | list[str],
        target: str | list[str],
    ) -> tuple[Float[Tensor, "optim_seq_len d_model"], list[list[list[float]]]]:
        """Runs the softprompt optimization.
        By default, trains a universal softprompt for all messages and targets.

        Args:
            messages: The messages to optimize on.
            target: The target generation.

        Returns:
            A tuple of the optimized embeddings and the epoch losses.
            Epoch losses are of shape losses[epoch][batch][step].
        """
        if isinstance(messages, str):
            messages = [messages]
        if isinstance(target, str):
            target = [target]

        message_dataloader = torch.utils.data.DataLoader(
            messages, batch_size=self.config.batch_size, shuffle=False
        )
        target_dataloader = torch.utils.data.DataLoader(
            target, batch_size=self.config.batch_size, shuffle=False
        )

        optim_ids, _ = tokenize(
            self.tokenizer, self.config.optim_str_init, add_special_tokens=False
        )
        optim_ids = optim_ids.to(self.device)
        optim_embeds = (
            self.model.get_input_embeddings()(optim_ids)
            .detach()
            .clone()
            .requires_grad_()
        )

        optimizer_eps = (
            # Use eps=1e-6 for bfloat16 and float16 or else we'll get NaNs
            1e-6 if self.model.dtype in [torch.float16, torch.bfloat16] else 1e-8
        )
        optimizer = torch.optim.Adam(
            [optim_embeds], lr=self.config.lr, eps=optimizer_eps
        )

        epoch_losses = []
        for epoch in range(self.config.num_epochs):
            batch_losses = []
            for messages, target in tqdm(
                zip(message_dataloader, target_dataloader),
                total=len(message_dataloader),
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.config.verbose,
            ):
                # Tokenizes and splits messages on <|optim_str|> token
                messages = add_optim_token_str_at_end(messages)
                input_ids, attn_mask = tokenize(
                    self.tokenizer,
                    messages,
                    add_chat_template=True,
                    padding="longest",
                    padding_side="left",
                )
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = (
                    split_tokenized_messages_on_optim_str(
                        input_ids, attn_mask, self.tokenizer.optim_token_id
                    )
                )

                target_ids, target_attn_mask = tokenize(
                    self.tokenizer,
                    target,
                    padding="longest",
                    padding_side="right",
                    add_special_tokens=False,
                )
                target_ids = target_ids.to(self.device)
                target_attn_mask = target_attn_mask.to(self.device)
                # Embeds everything that doesn't get optimized
                before_embeds, after_embeds, target_embeds = [
                    self.model.get_input_embeddings()(ids)
                    for ids in (left_input_ids, right_input_ids, target_ids)
                ]

                batch_size = input_ids.shape[0]
                losses = []
                for i in range(self.config.num_steps):
                    optimizer.zero_grad()

                    batched_optim_embeds = optim_embeds.expand(batch_size, -1, -1)
                    batched_attn_mask = torch.ones(
                        batch_size, optim_embeds.shape[1], device=self.device
                    )

                    input_embeds = torch.cat(
                        [
                            before_embeds.detach(),
                            batched_optim_embeds,
                            after_embeds.detach(),
                            target_embeds.detach(),
                        ],
                        dim=1,
                    )
                    input_attn_mask = torch.cat(
                        [
                            left_attn_mask.detach(),
                            batched_attn_mask,
                            right_attn_mask.detach(),
                            target_attn_mask.detach(),
                        ],
                        dim=1,
                    )

                    output = self.model(
                        inputs_embeds=input_embeds,
                        attention_mask=input_attn_mask,
                        output_hidden_states=True,
                    )
                    logits = output.logits

                    # Shift logits so token n-1 predicts token n
                    shift = input_embeds.shape[1] - target_ids.shape[1]
                    shift_logits = logits[..., shift - 1 : -1, :].contiguous()
                    # ^-- [batch_size, num_target_ids, vocab_size]
                    shift_labels = target_ids

                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss_float = loss.item()
                    losses.append(loss_float)

                    loss.backward()
                    optimizer.step()

                batch_losses.append(losses)
            epoch_losses.append(batch_losses)

        return optim_embeds.detach().cpu(), epoch_losses


def train_softprompt(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: str | list[str],
    target: str | list[str],
    config: SoftPromptConfig | None = None,
) -> tuple[Float[Tensor, "optim_seq_len d_model"], list[list[list[float]]]]:
    """Generates a single optimized string using soft-prompt optimization.
    By default, trains a universal softprompt for all messages and targets.

    Args:
        model: The model to optimize on.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The configuration to use.

    Returns:
        A tuple of the optimized embeddings and the epoch losses.
        Epoch losses are of shape losses[epoch][batch][step].
    """
    if config is None:
        config = SoftPromptConfig()

    logger.setLevel(logging.INFO if config.verbose else logging.WARNING)

    softprompt = SoftPrompt(model, tokenizer, config)
    optim_embeds, epoch_losses = softprompt.run(messages, target)
    return optim_embeds, epoch_losses
