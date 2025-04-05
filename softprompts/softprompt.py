import logging
from dataclasses import dataclass

import torch
import transformers
from jaxtyping import Int
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
    lr: float = 0.0001
    seed: int | None = None
    verbose: bool = True


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
        self.embedding_layer = self.model.get_input_embeddings()
        self.config = config
        self.device = model.device

        # add token <|optim_str|> for {optim_str} that will not be embedded
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.optim_str_token]}
        )
        self.optim_str_token_id = self.tokenizer.convert_tokens_to_ids(
            self.optim_str_token
        )
        self.optim_embeds = None

        # Ensure the tokenizer has a pad token
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

        if self.config.seed is not None:
            set_seed(self.config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

    def add_optim_str_token(self, messages: list[str]) -> list[str]:
        """Adds the <|optim_str|> token to each message."""
        return [x + self.optim_str_token for x in messages]

    def tokenize(
        self,
        text: str | list[str],
        add_chat_template: bool = False,
        **tokenizer_kwargs,
    ) -> tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch seq_len"]]:
        if isinstance(text, str):
            text = [text]

        if add_chat_template:
            # Assume that text argument is the user input
            batched_messages = [[{"role": "user", "content": msg}] for msg in text]
            text = self.tokenizer.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,  # type: ignore
            )
            if self.tokenizer.bos_token:
                # Remove the BOS token -- this will get added when tokenizing, if necessary
                text = [
                    text.replace(self.tokenizer.bos_token, "")
                    for text in text
                    if text.startswith(self.tokenizer.bos_token)
                ]

        inputs = self.tokenizer(
            text=text,
            return_tensors="pt",
            **tokenizer_kwargs,
        )

        return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(
            self.device
        )

    def split_tokenized_messages_on_optim_str(
        self,
        input_ids: Int[Tensor, "batch seq_len"],
        attn_mask: Int[Tensor, "batch seq_len"] | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Splits the tokenized messages on the <|optim_str|> token.
        Returns the left and right input ids and attention masks without the optim str token.
        If the attention mask is not provided, it is not returned.

        Args:
            input_ids: The tokenized messages.
            attn_mask: The attention mask. Optional, only returned if provided.
        """
        optim_str_positions = (input_ids == self.optim_str_token_id).nonzero(
            as_tuple=False
        )
        unique_columns = torch.unique(optim_str_positions[:, 1])
        counts = torch.sum(input_ids == self.optim_str_token_id, dim=1)
        assert torch.all(counts == 1), (
            "Not exactly one instance of self.optim_str_token_id in each row."
        )
        assert len(unique_columns) == 1, (
            "self.optim_str_token_id is not in the same column in all rows."
        )

        insertion_column = unique_columns.item()
        left_input_ids = input_ids[:, :insertion_column]
        right_input_ids = input_ids[:, insertion_column + 1 :]
        if attn_mask is not None:
            left_attn_mask = attn_mask[:, :insertion_column]
            right_attn_mask = attn_mask[:, insertion_column + 1 :]
            return left_input_ids, right_input_ids, left_attn_mask, right_attn_mask

        return left_input_ids, right_input_ids

    def generate_with_softprompt(
        self,
        messages: str | list[str],
        optim_embeds: Tensor,
        **generation_kwargs,
    ) -> str | list[str]:
        if isinstance(messages, str):
            messages = [messages]

        optim_embeds = optim_embeds.to(self.device)

        messages = self.add_optim_str_token(messages)
        input_ids, attn_mask = self.tokenize(
            messages,
            add_chat_template=True,
            padding="longest",
            padding_side="left",
        )
        left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = (
            self.split_tokenized_messages_on_optim_str(input_ids, attn_mask)
        )

        before_embeds, after_embeds = [
            self.embedding_layer(ids) for ids in (left_input_ids, right_input_ids)
        ]

        batched_optim_embeds = optim_embeds.expand(len(messages), -1, -1)
        batched_attn_mask = torch.ones(
            len(messages), optim_embeds.shape[1], device=self.device
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

        generation = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_attn_mask,
            **generation_kwargs,
        )

        return self.tokenizer.batch_decode(generation, skip_special_tokens=True)

    def run(
        self,
        messages: str | list[str],
        target: str | list[str],
    ) -> "SoftPrompt":
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

        optim_ids, _ = self.tokenize(
            self.config.optim_str_init, add_special_tokens=False
        )
        optim_embeds = self.embedding_layer(optim_ids).detach().clone().requires_grad_()

        optimizer = torch.optim.Adam([optim_embeds], lr=self.config.lr, eps=1e-6)

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
                messages = self.add_optim_str_token(messages)
                input_ids, attn_mask = self.tokenize(
                    messages,
                    add_chat_template=True,
                    padding="longest",
                    padding_side="left",
                )
                left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = (
                    self.split_tokenized_messages_on_optim_str(input_ids, attn_mask)
                )

                # Tokenizes target
                target_ids, target_attn_mask = self.tokenize(
                    target,
                    padding="longest",
                    padding_side="right",
                    add_special_tokens=False,
                )

                # Embeds everything that doesn't get optimized
                before_embeds, after_embeds, target_embeds = [
                    self.embedding_layer(ids)
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
                    shift_logits = logits[
                        ..., shift - 1 : -1, :
                    ].contiguous()  # [batch_size, num_target_ids, vocab_size]
                    shift_labels = target_ids

                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss_float = loss.item()
                    losses.append(loss_float)

                    # logger.info(f"Iter: {i:3d} | Loss: {loss_float}")

                    loss.backward()
                    optimizer.step()

                batch_losses.append(losses)
            epoch_losses.append(batch_losses)

        self.optim_embeds = optim_embeds.detach().cpu()
        return self


def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: str | list[str],
    target: str | list[str],
    config: SoftPromptConfig | None = None,
) -> SoftPrompt:
    """Generates a single optimized string using soft-prompt optimization.

    Args:
        model: The model to optimize on.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The configuration to use.

    Returns:
        A SoftPrompt object that contains the optimized strings.
    """
    if config is None:
        config = SoftPromptConfig()

    logger.setLevel(logging.INFO if config.verbose else logging.WARNING)

    softprompt = SoftPrompt(model, tokenizer, config)
    return softprompt.run(messages, target)
