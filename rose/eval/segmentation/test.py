import logging
from omegaconf import OmegaConf
import os
import sys
from typing import Any

PROJECT_ROOT = "/home/jiangwenjing/hd/rose-copy"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rose.eval.segmentation.config import SegmentationConfig
from rose.eval.helpers import args_dict_to_dataclass, cli_parser
from rose.eval.setup import load_model_and_context
from rose.run.init import job_context
from rose.eval.segmentation.run import run_segmentation_with_rose

logger = logging.getLogger("rose_segmentation_test")

def benchmark_launcher(test_argvs: dict[str, object]) -> dict[str, Any]:
    if "config" in test_argvs:
        base_config_path = test_argvs.pop("config")
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.normpath(os.path.join(PROJECT_ROOT, base_config_path))
        mode = test_argvs.pop("mode", "test")
        output_dir = test_argvs["output_dir"]
        base_config = OmegaConf.load(base_config_path)
        structured_config = OmegaConf.structured(SegmentationConfig)
        dataclass_config: SegmentationConfig = OmegaConf.to_object(
            OmegaConf.merge(
                structured_config,
                base_config,
                OmegaConf.create(test_argvs),
            )
        )
    else:
        dataclass_config, output_dir = args_dict_to_dataclass(eval_args=test_argvs, config_dataclass=SegmentationConfig)
    backbone = None
    if dataclass_config.model:
        backbone, _ = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    logger.info(f"Segmentation Config:\n{OmegaConf.to_yaml(dataclass_config)}")
    results_dict = run_segmentation_with_rose(backbone=backbone, config=dataclass_config, mode=mode)
    return results_dict
def main(argv=None):
    if argv == None:
        """argv = [
                "model.model_name=ROSE",
                "model.config_file=/data/wenjing/skin_dataset/rose-outputs/config.yaml",
                "model.pretrained_weights=/data/wenjing/skin_dataset/rose-outputs/eval/training_35999/teacher_checkpoint.pth",
                "config=rose/eval/segmentation/configs/config-ham10k-linear-seg-training.yaml",
                "datasets.root=/data/wenjing/skin_dataset/ssl/segmentation/HAM10000",
                "load_from=/data/wenjing/skin_dataset/rose-outputs/segmentation/ham10k_rose_linear_seg_output/model_final.pth",
                "output_dir=/data/wenjing/skin_dataset/rose-outputs/segmentation/ham10k_rose_linear_seg_output",
        ]"""
        argv = sys.argv[1:]
    test_argvs = cli_parser(argv)
    with job_context(output_dir=test_argvs["output_dir"]):
        test_argvs["mode"] = "test"
        benchmark_launcher(test_argvs)
    return 0


if __name__ == "__main__":
    sys.exit(main())