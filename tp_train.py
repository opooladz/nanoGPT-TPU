"""
This training script updates NanoGPT to run with either FSDP, or FSDP + Tensor Parallel (2D).
Usage:
a - ensure run_tp.sh has the right amount of gpus (_nproc_per_node = your gpu count)
b - you can control FSDP vs FSDP + TP using config/config_2D.py, setting the 
    use_tensor_parallel: bool = True/False
c - to run, bash run_tp.sh

"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.distributed.tensor.parallel._utils import _create_1d_device_mesh
import inspect

import config.config_2D as fsdp_config

cfg = fsdp_config.train_config()

TP_AVAILABLE = False
try:
    from torch.distributed._tensor import (
        DeviceMesh,
    )
    from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
        # get_parallelization_fqn,
    )

    # need to setup hooks for TP

    TP_AVAILABLE = enable_2d_with_fsdp()

except BaseException as e:
    print(f"Exception during TP init - {e=}\n")
    pass

assert TP_AVAILABLE, f"fsdp did not init"


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 2
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "openwebtext"
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)

compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup

# Init TP
_tp = int(os.environ.get("RANK", -1)) != -1  # verify distributed run

assert (
    _tp and TP_AVAILABLE
), "this config assumes setup for Tensor Parallel - distributed not ready here."


init_process_group(backend=backend)
_rank = int(os.environ["RANK"])
_local_rank = int(os.environ["LOCAL_RANK"])

world_size = int(os.environ["WORLD_SIZE"])  # total number of training processes
device = f"cuda:{_local_rank}"
torch.cuda.set_device(device)
master_process = _rank == 0  # this process will do logging, checkpointing etc.
seed_offset = _rank  # each process gets a different seed


# wrapper to avoid cluttering with if rank==0...
def rank_print(x):
    if _rank == 0:
        print(x)


rank_print(f"TP is available = {TP_AVAILABLE}\n")
model_parallel_size = 2

# 2-D mesh is [dp, tp]
twod_mesh = DeviceMesh(
    device_type="cuda",
    mesh=torch.arange(0, world_size).view(-1, model_parallel_size),
)
rank_print(f"{twod_mesh=}")


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

dtype = "bfloat16"  # 'float32', 'bfloat16'
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]


# poor man's data loader
data_dir = os.path.join(cfg.data_dir, cfg.dataset)
rank_print(f"{data_dir=}")
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split, fsdp_pg):
    data = train_data if split == "train" else val_data
    # Training data needs to be same across TP ranks.
    torch.manual_seed(dist.get_rank(fsdp_pg))
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init


# if init_from == "scratch":
# init a new model from scratch
rank_print("Initializing a new model from scratch")

# _pure_fsdp = True
_2D = cfg.use_tensor_parallel


if _2D:
    tp_device_mesh = _create_1d_device_mesh(twod_mesh, -1)
    
else:
    tp_device_mesh = None
model, model_config = fsdp_config.build_model(cfg, tp_device_mesh)
model.cuda(_rank)


import torch.distributed as dist
import torch.nn as nn


if _2D:
    from tp_handlers import parallelize_model

    num_layers = parallelize_model(model, model_config, twod_mesh)

    
    rank_print(f"initialized model for 2D with {num_layers} layers.\n")

if _2D:
    fsdp_pg = twod_mesh.get_dim_groups()[0]
else:
    fsdp_pg = None

# todo - add back main code later for resume
mixed_precision_policy = fsdp_config.set_mixed_precision_policy()

model = FSDP(
    model, 
    auto_wrap_policy=cfg.wrapping_policy, 
    mixed_precision = mixed_precision_policy, 
    device_id=device, 
    process_group=fsdp_pg
)


# optimizer
# new PyTorch nightly has a new 'fused' option for AdamW that is much faster

use_fused = (device_type == "cuda") and (
    "fused" in inspect.signature(torch.optim.AdamW).parameters
)
print(f"using fused AdamW: {use_fused}")
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **extra_args)
# optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

# optimizer = model.configure_optimizers(
#    weight_decay, learning_rate, (beta1, beta2), device_type
# )
# if init_from == "resume":
#    optimizer.load_state_dict(checkpoint["optimizer"])

# compile the model
"""if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
"""


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(fsdp_pg):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, fsdp_pg)
            print("after get batch ", k)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
X, Y = get_batch("train", fsdp_pg)  # fetch the very first batch

local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
eval_interval = 1
warmup = 5

while local_iter_num < cfg.iters_to_run:
    t0 = time.time()
    logits, loss = model(X, Y)
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch("train", fsdp_pg)
    # backward pass, with gradient scaling if training in fp16
    loss.backward()
    # clip the gradient
    # if grad_clip != 0.0:
    #    scaler.unscale_(optimizer)
    #    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    optimizer.step()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0

    if iter_num >= warmup:
        lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
        # if local_iter_num >= 3:  # let the training loop settle a bit
        mfu = model.estimate_mfu(
            batch_size * world_size * gradient_accumulation_steps, dt
        )
        running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        rank_print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1
    rank_print(f"iter {iter_num} completed...")

    # termination conditions
    if iter_num > max_iters:
        break

        # determine and set the learning rate for this iteration
        # lr = get_lr(iter_num) if decay_lr else learning_rate
        # for param_group in optimizer.param_groups:
        #    param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints

        # Running the estimate_loss on rank0 only will cause hang because it calls into a distributed model!
        # if iter_num % eval_interval == 0 and master_process and False:
        # losses = estimate_loss(fsdp_pg)
        # print(
        #    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        # )

        """if losses["val"] < best_val_loss or always_save_checkpoint:
        best_val_loss = losses["val"]
        raw_model = model.module if _tp else model
        if iter_num > 0:
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        """
        # if iter_num == 0 and eval_only:
        #    break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    """for micro_step in range(gradient_accumulation_steps):
        if _tp or _tp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
        """

dist.barrier()
rank_print(
    f"Training completed.  \nRun used tensor_parallel = {cfg.use_tensor_parallel}"
)
destroy_process_group()