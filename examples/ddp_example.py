#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse
import uuid
from typing import Optional

import torch
import torchsnapshot
from torchsnapshot.snapshot import Snapshot
from torchsnapshot.stateful import AppState

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


NUM_EPOCHS = 4
EPOCH_SIZE = 16
BATCH_SIZE = 8


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size, args):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    model = Model().to(rank)

    # DDP wrapper around model: create model and move it to GPU with id rank
    ddp_model = DDP(model, device_ids=[rank])

    # optim = torch.optim.Adagrad(model.parameters(), lr=0.01)
    # TODO: does this make a difference?
    optim = torch.optim.Adagrad(ddp_model.parameters(), lr=0.01)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    progress = torchsnapshot.StateDict(current_epoch=0)

    # torchsnapshot: define app state
    app_state: AppState = {
        "rng_state": torchsnapshot.RNGState(),
        "model": model,
        "optim": optim,
        "progress": progress,
    }
    snapshot: Optional[Snapshot] = None

    while progress["current_epoch"] < NUM_EPOCHS:
        # torchsnapshot: restore app state
        if snapshot is not None:
            snapshot.restore(app_state)

        for _ in range(EPOCH_SIZE):
            X = torch.rand((BATCH_SIZE, 128))
            # label = torch.rand((BATCH_SIZE, 1))
            # pred = model(X)
            pred = ddp_model(X)
            label = torch.rand((BATCH_SIZE, 1)).to(rank)

            loss = loss_fn(pred, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # Do we need to clean up everytime? will for loop through epochs here be a concern?

        progress["current_epoch"] += 1

        # torchsnapshot: take snapshot
        snapshot = torchsnapshot.Snapshot.take(
            f"{args.work_dir}/{uuid.uuid4()}", app_state
        )

        print(f"Snapshot path: {snapshot.path}")
    cleanup()


def run_demo(demo_fn, world_size, args):
    mp.spawn(demo_fn, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/tmp")
    args: argparse.Namespace = parser.parse_args()

    torch.random.manual_seed(42)

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size, args)
