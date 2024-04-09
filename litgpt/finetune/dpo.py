# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import dataclasses
import math
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F

from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from torch.utils.data import DataLoader
from torchmetrics import RunningMean

from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca, DataModule
from litgpt.generate.base import generate
from litgpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from litgpt.prompts import save_prompt_style
from litgpt.scripts.merge_lora import merge_lora
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CLI,
    CycleIterator,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)

import transformers
from transformers import AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

def setup(
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/finetune/lora"),
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: Union[int, str] = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = True,
    lora_value: bool = True,
    lora_projection: bool = True,
    lora_mlp: bool = True,
    lora_head: bool = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=16,
        micro_batch_size=4,
        lr_warmup_steps=100,
        epochs=5,
        learning_rate=5e-4,
        max_seq_length=None,
    ),
    eval: EvalArgs = EvalArgs(interval=100, max_new_tokens=100, max_iters=100),
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 1337,
) -> None:
    """Finetune a model using the LoRA method.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        quantize: If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information.
        devices: How many devices/GPUs to use.
        lora_r: The LoRA rank.
        lora_alpha: The LoRA alpha.
        lora_dropout: The LoRA dropout value.
        lora_query: Whether to apply LoRA to the query weights in attention.
        lora_key: Whether to apply LoRA to the key weights in attention.
        lora_value: Whether to apply LoRA to the value weights in attention.
        lora_projection: Whether to apply LoRA to the output projection in the attention block.
        lora_mlp: Whether to apply LoRA to the weights of the MLP in the attention block.
        lora_head: Whether to apply LoRA to output head in GPT.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """

    pprint(locals())
    data = Alpaca() if data is None else data
    devices = parse_devices(devices)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(
        checkpoint_dir / "model_config.yaml",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_query=lora_query,
        lora_key=lora_key,
        lora_value=lora_value,
        lora_projection=lora_projection,
        lora_mlp=lora_mlp,
        lora_head=lora_head,
    )

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(logger_name, out_dir, name=f"finetune-{config.name}", log_interval=train.log_interval)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    if devices > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)
    fabric.launch(main, devices, seed, config, data, checkpoint_dir, out_dir, train, eval)


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
) -> None:
    validate_args(train, eval)

    #tokenizer = Tokenizer(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices)
    lr_max_steps = min(train.epochs * steps_per_epoch, (train.max_steps or float("inf")))

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    
    with fabric.init_module(empty_init=(devices > 1)):
        ref_model = GPT(config)
        model = GPT(config)

    disable_dropout(model)
    disable_dropout(ref_model)
    
    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}")

    
    ref_model = fabric.setup_module(ref_model)

    #trainable_params = [p for p in model.parameters() if p.requires_grad]

    '''
    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.PagedAdamW
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_params, lr=train.learning_rate, weight_decay=train.weight_decay, betas=(train.beta1, train.beta2)
    )
    optimizer = fabric.setup_optimizers(optimizer)
    '''
    optimizer = torch.optim.RMSprop(model.parameters(), lr=train.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps)

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)
    load_checkpoint(fabric, ref_model, checkpoint_path, strict=False)
    
    ref_model = ref_model.eval()

    train_time = time.perf_counter()
    fit(
        fabric,
        model,
        ref_model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        devices,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        data,
    )

    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "final" / "lit_model.pth.lora"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_lora_checkpoint(fabric, model, save_path)
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_path.parent)
        save_hyperparameters(setup, save_path.parent)
        save_prompt_style(data.prompt_style, save_path.parent)
        merge_lora(checkpoint_dir=save_path.parent)

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch



def fit(
    fabric: L.Fabric,
    model: GPT,
    ref_model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    #longest_seq_length, longest_seq_ix = get_longest_seq_length(train_dataloader.dataset)

    #tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    ref_model.eval()

    longest_seq_length = 512
    model.max_seq_length = train_dataloader.dataset.max_seq_length #min(longest_seq_length, train.max_seq_length or float("inf"))
    
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(fabric, model, ref_model, val_dataloader, tokenizer, dataclasses.replace(eval, max_iters=2), data)  # sanity check

    train_iterator = CycleIterator(train_dataloader)
    throughput = ThroughputMonitor(fabric, window_size=50)
    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    max_steps = train.max_steps or float("inf")
    step_count = 0
    iter_num = 0
    total_lengths = 0
    total_t0 = time.perf_counter()
    
    val_loss = "n/a"
    
    loss_config = {"name": "dpo",
                   "beta":0.1,
                   "label_smoothing":0,
                   "reference_free":False,    
                   }
    
    while step_count < max_steps and train_iterator.epoch < train.epochs:
        iter_num += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)

        is_accumulating = iter_num % train.gradient_accumulation_iters(devices) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
         
            loss, metrics = get_batch_metrics(fabric, model, ref_model, batch, loss_config, train=True)

            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())
        
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step_count += 1

        total_lengths += batch['chosen_input_ids'].numel()
        if iter_num % train.log_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0, batches=iter_num, samples=iter_num * train.micro_batch_size, lengths=total_lengths
            )
            throughput.compute_and_log(step=iter_num)
            metrics = {
                "loss": loss,
                "iter": iter_num,
                "step": step_count,
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": iter_num * train.micro_batch_size * model.config.block_size,
                "total_tokens": (iter_num * train.micro_batch_size * model.config.block_size * fabric.world_size),
                "learning_rate": scheduler.get_last_lr()[0],
                **metrics
            }
            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=iter_num)

        if not is_accumulating and step_count % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, ref_model, val_dataloader, tokenizer, eval, data)
            t1 = time.perf_counter() - t0
            fabric.print(f"iter {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=iter_num)
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and step_count % train.save_interval == 0:
            checkpoint_file = out_dir / f"step-{step_count:06d}" / "lit_model.pth.lora"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            save_lora_checkpoint(fabric, model, checkpoint_file)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data.prompt_style, checkpoint_file.parent)


def concatenated_forward(model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
    
        We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    concatenated_batch = concatenated_inputs(batch)
    #all_logits = model(concatenated_batch['concatenated_input_ids'],
    #                   attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
    all_logits = model(concatenated_batch['concatenated_input_ids']).to(torch.float32)
    all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
    chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
    rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
    return chosen_logps, rejected_logps

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def get_batch_metrics(fabric, policy, reference_model, batch: Dict[str, Union[List, torch.LongTensor]], loss_config, train=True):
    """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

    metrics = {}
    train_test = 'train' if train else 'eval'

    if loss_config['name'] in {'dpo', 'ipo'}:
        policy_chosen_logps, policy_rejected_logps = concatenated_forward(policy, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = concatenated_forward(reference_model, batch)

        if loss_config['name'] == 'dpo':
            loss_kwargs = {'beta': loss_config['beta'], 'reference_free': loss_config['reference_free'], 'label_smoothing': loss_config['label_smoothing'], 'ipo': False}
        elif loss_config['name'] == 'ipo':
            loss_kwargs = {'beta': loss_config['beta'], 'ipo': True}
        else:
            raise ValueError(f'unknown loss {loss_config["name"]}')

        losses, chosen_rewards, rejected_rewards = preference_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        chosen_rewards = fabric.all_gather(chosen_rewards) #all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
        rejected_rewards = fabric.all_gather(rejected_rewards) #all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
        reward_accuracies = fabric.all_gather(reward_accuracies) #all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

        metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

        policy_rejected_logps = fabric.all_gather(policy_rejected_logps.detach()) #all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

    elif loss_config['name'] == 'sft':
        policy_chosen_logits = policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
        policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

        losses = -policy_chosen_logps

    policy_chosen_logps = fabric.all_gather(policy_chosen_logps.detach()) #all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
    metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

    all_devices_losses = fabric.all_gather(losses.detach()) #all_gather_if_needed(losses.detach(), self.rank, self.world_size)
    metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

    return losses.mean(), metrics



# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: GPT,
    ref_model,
    val_dataloader: DataLoader,
    tokenizer: Tokenizer,
    eval: EvalArgs,
    data: DataModule
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    loss_config = {"name": "dpo",
                   "beta":0.1,
                   "label_smoothing":0,
                   "reference_free":False}

    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break

        losses, eval_metrics = get_batch_metrics(fabric, model, ref_model, batch, loss_config, train=False)
        
        #input_ids, targets = batch["input_ids"], batch["labels"]
        #logits = model(input_ids)
        #losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)

    val_loss = losses.mean()

    '''
    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    prompt = data.prompt_style.apply(instruction)
    print(prompt)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)

    output = generate(
        model, encoded, max_returned_tokens=len(encoded) + eval.max_new_tokens, temperature=0.8, eos_id=tokenizer.eos_id
    )
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    fabric.print(output)
    '''
    model.train()

    return val_loss


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    #scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    #scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    #return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=(lambda step: min(1.0, (step + 1) / (warmup_steps + 1)))
    )


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs
) -> Tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=train.max_seq_length)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    return train_dataloader, val_dataloader


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    print(data)
    lengths = [len(d["prompt"]) + len(d["chosen"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_lora_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if not train.epochs and not train.max_steps:
        issues.append(f"{__file__} requires either epochs or max_steps to be set. This is set in {train}")
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(setup)
