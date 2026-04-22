import argparse
from elefant.policy_model.config import LightningPolicyConfig
from elefant.config import load_config
from elefant.policy_model.stage3_finetune import train_stage3_finetune, _select_training_gpus
import logging
import os
import sys
import torch
import elefant.torch
from elefant.torch import configure_logging


def _maybe_reexec_with_filtered_gpus():
    if os.environ.get("ELEFANT_STAGE3_GPU_FILTERED"):
        return
    accelerator, _devices = _select_training_gpus()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if accelerator != "gpu" or not visible:
        return
    env = os.environ.copy()
    env["ELEFANT_STAGE3_GPU_FILTERED"] = "1"
    logging.warning(
        "Re-launching training with CUDA_VISIBLE_DEVICES=%s so Lightning starts with the filtered GPU set.",
        visible,
    )
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def lightning_main():
    configure_logging(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--data_folder", type=str, default=None)
    parser.add_argument(
        "--resume_from_ckpt",
        type=str,
        default=None,
        help="Checkpoint file or directory to resume stage3 training from.",
    )
    parser.add_argument(
        "--init_from_origin_ckpt",
        type=str,
        default=None,
        help="Checkpoint file or directory from /data/jsy/open-p2p_origin used for partial initialization before training.",
    )
    args = parser.parse_args()

    _maybe_reexec_with_filtered_gpus()

    config = load_config(args.config, LightningPolicyConfig)
    config.shared.fast_dev_run = args.fast_dev_run

    if args.data_folder is not None:
        logging.info(f"Using local data folder: {args.data_folder}")
        config.stage3_finetune.training_dataset.local_prefix = args.data_folder
        config.stage3_finetune.training_dataset.dataset_path = args.data_folder
        for val_dataset in config.stage3_finetune.validation_datasets:
            val_dataset.local_prefix = args.data_folder
            val_dataset.dataset_path = args.data_folder

    if args.resume_from_ckpt is not None:
        logging.info(f"Using resume checkpoint: {args.resume_from_ckpt}")
        config.stage3_finetune.init.stage3_model_path = args.resume_from_ckpt

    if args.init_from_origin_ckpt is not None:
        logging.info(f"Using origin initialization checkpoint: {args.init_from_origin_ckpt}")
        config.stage3_finetune.init.origin_model_path = args.init_from_origin_ckpt

    if args.fast_dev_run:
        logging.warning("!!!Fast dev run is enabled!!!")

    if args.no_compile:
        logging.warning("!!!No compile is enabled!!!")
        torch.compiler.set_stance("force_eager")

    elefant.torch.pytorch_setup()
    train_stage3_finetune(config)


if __name__ == "__main__":
    lightning_main()
