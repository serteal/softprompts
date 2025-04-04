import logging
from dataclasses import dataclass
from typing import Union

import torch
import transformers
from jaxtyping import Float, Int
from torch import Tensor
from transformers import set_seed

logger = logging.getLogger("nanosoftprompt")
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
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    lr: float = 0.0001
    seed: int | None = None
    verbosity: str = "INFO"


@dataclass
class SoftPromptResult:
    losses: list[float]
    optim_embeds: Tensor


class SoftPrompt:
    optim_str_token = "<|optim_str|>"

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: SoftPromptConfig,
    ):
        self.model = model
        for param in model.parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device

        # add token <|optim_str|> for {optim_str} that will not be embedded
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.optim_str_token]}
        )
        self.optim_str_token_id = self.tokenizer.convert_tokens_to_ids(
            self.optim_str_token
        )

        if self.config.seed is not None:
            set_seed(self.config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

    def normalize_messages(
        self, messages: str | list[str] | list[dict] | list[list[dict]]
    ) -> list[list[dict]]:
        if isinstance(messages, str):
            return [[{"role": "user", "content": messages}]]
        elif isinstance(messages, list) and isinstance(messages[0], str):
            return [[{"role": "user", "content": m}] for m in messages]
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            return [messages]
        elif (
            isinstance(messages, list)
            and isinstance(messages[0], list)
            and isinstance(messages[0][0], dict)
        ):
            return messages
        else:
            raise ValueError(
                f"Invalid messages type: {type(messages)}. "
                "Expected str, list[str], list[dict], or list[list[dict]]"
            )

    def add_optim_token(self, dialog_list: list[list[dict]]) -> list[list[dict]]:
        for dialog in dialog_list:
            if not any([self.optim_str_token in d["content"] for d in dialog]):
                dialog[-1]["content"] = dialog[-1]["content"] + self.optim_str_token
        return dialog_list

    def tokenize_and_split_messages(
        self, messages: list[list[dict]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # adds optim_str_token to the last message in each dialog, noop if already present
        messages = self.add_optim_token(messages)
        templates = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        logger.debug(templates)

        full_tokenized_templates = self.tokenizer(
            templates,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
        )
        full_input_ids = full_tokenized_templates["input_ids"]
        full_attn_mask = full_tokenized_templates["attention_mask"]
        # ^-- [batch_size, seq_len]

        # check the optim str token is in the same position for all templates
        optim_str_positions = (full_input_ids == self.optim_str_token_id).nonzero(
            as_tuple=False
        )
        unique_columns = torch.unique(optim_str_positions[:, 1])
        counts = torch.sum(full_input_ids == self.optim_str_token_id, dim=1)
        assert torch.all(counts == 1), (
            "Not exactly one instance of self.optim_str_token_id in each row."
        )
        assert len(unique_columns) == 1, (
            "self.optim_str_token_id is not in the same column in all rows."
        )

        insertion_column = unique_columns.item()
        left_input_ids = full_input_ids[:, :insertion_column]
        right_input_ids = full_input_ids[:, insertion_column + 1 :]
        left_attn_mask = full_attn_mask[:, :insertion_column]
        right_attn_mask = full_attn_mask[:, insertion_column + 1 :]

        return left_input_ids, right_input_ids, left_attn_mask, right_attn_mask

    def run(
        self,
        messages: str | list[str] | list[dict] | list[list[dict]],
        target: str | list[str],
    ) -> SoftPromptResult:
        messages = self.normalize_messages(messages)
        left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = (
            self.tokenize_and_split_messages(messages)
        )

        if isinstance(target, str):
            target = [target]
        tokenized_target = self.tokenizer(
            target,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
            padding_side="right",
        )
        target_ids = tokenized_target["input_ids"]
        target_attn_mask = tokenized_target["attention_mask"]

        # Embed everything that doesn't get optimized
        embedding_layer = self.model.get_input_embeddings()
        before_embeds, after_embeds, target_embeds = [
            embedding_layer(ids)
            for ids in (left_input_ids, right_input_ids, target_ids)
        ]

        # # Compute the KV Cache for tokens that appear before the optimized tokens
        # with torch.no_grad():
        #     output = self.model(inputs_embeds=before_embeds, use_cache=True)
        #     prefix_cache = output.past_key_values

        optim_ids = self.tokenizer(
            self.config.optim_str_init,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        optim_embeds = embedding_layer(optim_ids).detach().clone().requires_grad_()

        optimizer = torch.optim.Adam([optim_embeds], lr=self.config.lr, eps=1e-6)

        losses = []
        for i in range(self.config.num_steps):
            optimizer.zero_grad()
            batched_optim_embeds = optim_embeds.expand(left_input_ids.shape[0], -1, -1)
            batched_attn_mask = torch.ones(left_input_ids.shape[0], optim_ids.shape[1])
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
            shift_logits = logits[
                ..., shift - 1 : -1, :
            ].contiguous()  # [batch_size, num_target_ids, vocab_size]
            shift_labels = target_ids

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss_float = loss.item()
            losses.append(loss_float)

            logger.debug(f"Iter: {i:3d} | Loss: {loss_float}")

            loss.backward()
            optimizer.step()

        result = SoftPromptResult(
            losses=losses,
            optim_embeds=optim_embeds.cpu(),
        )

        return result


def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, list[dict]],
    target: str,
    config: SoftPromptConfig | None = None,
) -> SoftPromptResult:
    """Generates a single optimized string using soft-prompt optimization.

    Args:
        model: The model to optimize on.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The configuration to use.

    Returns:
        A SoftPromptResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = SoftPromptConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    softprompt = SoftPrompt(model, tokenizer, config)
    return softprompt.run(messages, target)
