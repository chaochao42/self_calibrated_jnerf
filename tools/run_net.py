from ast import parse
import jittor as jt
from tqdm import tqdm
import argparse
import numpy as np
import os
from jnerf.runner import Runner,NeuSRunner, NeuS_Trainable_Runner, NeuS_Trainable_Freezed_Runner, NeuS_Trainable_Barf_Runner
from jnerf.utils.config import init_cfg, get_cfg
from jnerf.utils.registry import build_from_cfg,NETWORKS,SCHEDULERS,DATASETS,OPTIMS,SAMPLERS,LOSSES
# jt.flags.gopt_disable=1
jt.flags.use_cuda = 1
# jt.flags.lazy_execution=0

def main():
    assert jt.flags.cuda_archs[0] >= 61, "Failed: Sm arch version is too low! Sm arch version must not be lower than sm_61!"
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
    )

    parser.add_argument(
        "--type",
        default="novel_view",
        type=str,
    )
    parser.add_argument(
        "--mcube_threshold",
        default=0.0,
        type=float,
    )
    
    args = parser.parse_args()

    assert args.type in ["novel_view","mesh", "self_calibrated", "self_calibrated_freezed", "self_calibrated_barf"],f"{args.type} not support, please choose [novel_view, mesh]"
    assert args.task in ["train","test","render", "validate_mesh", "generate_depth"],f"{args.task} not support, please choose [train, test, render, validate_mesh]"
    
    if args.task == 'validate_mesh':
        is_continue = True

    if args.config_file:
        init_cfg(args.config_file)

    if args.type == 'novel_view':
        runner = Runner()
    elif args.type == 'mesh':
        runner = NeuSRunner() #is_continue=is_continue)
    elif args.type == 'self_calibrated':
        runner = NeuS_Trainable_Runner()
    elif args.type == 'self_calibrated_freezed':
        runner = NeuS_Trainable_Freezed_Runner()
    elif args.type == 'self_calibrated_barf':
        runner = NeuS_Trainable_Barf_Runner()
    else:
        print('Not support yet!')

    if args.task == "train":
        runner.train()
    elif args.task == "test":
        runner.test(True)
    elif args.task == "render":
        runner.render(True, args.save_dir)
    elif args.task == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold, name="validate", auto_load=True)
    elif args.task == 'generate_depth':
        runner.validate_debug_image(idx=0, auto_load=True)

if __name__ == "__main__":
    main()
