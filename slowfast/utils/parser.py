#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# My parser file
"""Argument parser functions."""

import argparse
import sys

from .def_config import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # My work for more command line arguments for wandb sweep
    parser.add_argument(
        "--solver_base_lr",
        help="Base learning rate for the solver",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--solver_lr_gamma",
        help="Learning rate decay gamma for the solver",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--solver_max_epochs",
        help="Maximum number of epochs for the solver",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--solver_weight_decay",
        help="Weight decay for the solver",
        default=0.0005,
        type=float,
    )
    parser.add_argument(
        "--train_batch_size",
        help="Batch size for training",
        default=8,
        type=int,
    )


    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # My work for more command line arguments for wandb sweep
    if hasattr(args, "solver_base_lr"):
        cfg.SOLVER.BASE_LR = args.solver_base_lr
    if hasattr(args, "solver_lr_gamma"):
        cfg.SOLVER.LR_GAMMA = args.solver_lr_gamma
    if hasattr(args, "solver_max_epochs"):
        cfg.SOLVER.MAX_EPOCHS = args.solver_max_epochs
    if hasattr(args, "solver_weight_decay"):
        cfg.SOLVER.WEIGHT_DECAY = args.solver_weight_decay
    if hasattr(args, "train_batch_size"):
        cfg.TRAIN.BATCH_SIZE = args.train_batch_size

    return cfg
