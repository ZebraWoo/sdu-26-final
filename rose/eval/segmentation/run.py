# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from omegaconf import OmegaConf
import os
import sys
from typing import Any

PROJECT_ROOT = "/home/jiangwenjing/hd/rose-copy"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rose.eval.segmentation.config import SegmentationConfig
from rose.eval.segmentation.eval import test_segmentation
from rose.eval.segmentation.train import train_segmentation
from rose.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from rose.eval.setup import load_model_and_context
from rose.run.init import job_context


logger = logging.getLogger("rose_segmentation_eval")

RESULTS_FILENAME = "results-semantic-segmentation.csv"
MAIN_METRICS = ["mIoU"]

# def run_segmentation_with_rose(
#     backbone,
#     config,
#     mode,
# ):
#     if config.load_from:
#         logger.info("Testing model performance on a pretrained decoder head")
#         return test_segmentation(backbone=backbone, config=config, mode=mode)
#     assert config.decoder_head.type == "linear", "Only linear head is supported for training"
#     return train_segmentation(backbone=backbone, config=config)

def run_segmentation_with_rose(
    backbone,
    config,
    mode,
):
    if config.load_from:
        logger.info("Testing model performance on a pretrained decoder head")
        return test_segmentation(backbone=backbone, config=config, mode=mode)
    # 允许 linear 和 m2f 两种头
    if config.decoder_head.type in ["linear", "m2f"]:
        return train_segmentation(backbone=backbone, config=config)
    raise ValueError(f"Unsupported decoder_head.type={config.decoder_head.type}")


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    if "config" in eval_args:  # using a config yaml file, useful for training
        base_config_path = eval_args.pop("config")
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.normpath(os.path.join(PROJECT_ROOT, base_config_path))
        mode = eval_args.pop("mode", "train")
        output_dir = eval_args["output_dir"]
        base_config = OmegaConf.load(base_config_path)
        structured_config = OmegaConf.structured(SegmentationConfig)
        dataclass_config: SegmentationConfig = OmegaConf.to_object(
            OmegaConf.merge(
                structured_config,
                base_config,
                OmegaConf.create(eval_args),
            )
        )
    else:  # either using default values, or only adding some args to the command line
        dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=SegmentationConfig)
    backbone = None
    if dataclass_config.model:
        backbone, _ = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    else:
        assert dataclass_config.load_from == "dinov3_vit7b16_ms"
    logger.info(f"Segmentation Config:\n{OmegaConf.to_yaml(dataclass_config)}")
    segmentation_file_path = os.path.join(output_dir, "segmentation_config.yaml")
    OmegaConf.save(config=dataclass_config, f=segmentation_file_path)
    results_dict = run_segmentation_with_rose(backbone=backbone, config=dataclass_config, mode=mode)
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        """argv = [
            # Hydra / OmegaConf style overrides
            "model.model_name=ROSE",
            "model.config_file=/data/wenjing/skin_dataset/rose-outputs/config.yaml",
            "model.pretrained_weights=/data/wenjing/skin_dataset/rose-outputs/eval/training_35999/teacher_checkpoint.pth",
            "config=rose/eval/segmentation/configs/config-isic2018-linear-training.yaml",
            "datasets.root=/data/wenjing/skin_dataset/ssl/segmentation/isic2018",
            "output_dir=/data/wenjing/skin_dataset/rose-outputs/segmentation/isic2018_linear_seg_output",
        ]"""
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    with job_context(output_dir=eval_args["output_dir"]):
        eval_args["mode"] = "train"
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
