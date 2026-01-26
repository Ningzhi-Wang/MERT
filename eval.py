import argparse
import os
import yaml
import json
import boto3
import numpy as np
import torch
import pytorch_lightning as pl
from itertools import product
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from pathlib import Path

from mert_fairseq.models.mert import MERTConfig, MERTModel
from mert_fairseq.tasks.mert_pretraining import MERTPretrainingTask
from fairseq import checkpoint_utils
from musiceval.datasets import LatentModule
from musiceval import tasks


TASK_PATH = {
    "MTT": "c4dm-datasets/MagnaTagATune/",
    "GTZAN": "c4dm-datasets/gtzan/",
    "EMO": "c4dm-datasets/emomuisc/",
    "Giantsteps": "c4dm-datasets/Giantsteps/",
    "NSynthPitch": "c4dm-datasets/NSynth/",
    "NSynthTimbre": "c4dm-datasets/NSynth/",
    "MEDLEYDB": "/data/scratch/acw713/datasets/medleydb/V1/",
    "MTGGenre": "/data/scratch/acw713/datasets/mtg/",
    "MTGInstrument": "/data/scratch/acw713/datasets/mtg/",
    "PhonationModes": "/data/scratch/acw713/datasets/phonation_modes",
}



def evaluate(config, ckpt):
    assert ckpt is not None, "Checkpoint must be provided for evaluation"
    task_names = config["task_names"]

    # Download checkpoint from S3 if not present locally
    save_dir  = config['trainer']['path']
    if not os.path.isfile(ckpt):
        os.makedirs(save_dir, exist_ok=True)
        aws_key = os.environ.get("AWS_ACCESS_KEY_C4DM02")
        aws_secret = os.environ.get("AWS_SECRET_KEY_C4DM02")
        s3 = boto3.client(
            "s3",
            endpoint_url="https://ceph-private-object-rgw.comp-research.qmul.ac.uk",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
        )
        s3.download_file(
            Bucket="c4dm-02",
            Key=ckpt,
            Filename=os.path.join(save_dir, "checkpoint.ckpt"),
        )
        ckpt = os.path.join(save_dir, "checkpoint.ckpt")

    # Perform hyperparameter search as in MERT
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    batch_size = [64]
    drop_outs = [0.2]
    l2s = [0]

    train_sizes = [-1]

    settings = list(product(lrs, batch_size, drop_outs, l2s))
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([ckpt])
    encoder = models[0]
    encoder.eval()

    config["trainer"].pop("epochs", None)
    root_path = config["trainer"].pop("path", None)
    assert root_path is not None, "Root path must be provided for evaluation"

    for task_name in task_names:
        for train_size in train_sizes:
            results = {}
            task_config = config.copy()
            task = eval(f"tasks.{task_name}")(TASK_PATH[task_name], **task_config["task"])
            data_param = task_config["data"]
            model_param = task_config["model"]
            trainer_param = task_config["trainer"]
            trainer_param["default_root_dir"] = root_path
            data_param["trainset_size"] = train_size
            data_param['encoder'] = encoder
            # datamodule = LatentModule(**data_param)
            datamodule = task.dataset(**data_param)
            datamodule.setup("fit")
            datamodule.setup("test")
            for setting in settings:
                print(f"evaluating {task_name} with setting: {setting}")
                log_path = Path(root_path)
                # wandb_logger = get_logger(log_path.stem, str(log_path / f"logs/{task}"))
                model_param["lr"] = setting[0]
                model_param["dropout"] = setting[2]
                model_param["l2"] = setting[3]
                model_param["feature_loaded"] = datamodule.feature_loaded
                model = task.prober(**model_param)
                early_stop = EarlyStopping(
                    monitor=f"{task_name}_val_loss",
                    patience=20,
                    mode="max",
                    verbose=True,
                )
                checkpoint_callback = ModelCheckpoint(
                    dirpath=log_path / task_name,
                    filename=f"{task_name}-best-checkpoint-{'-'.join(map(str, setting))}",
                    save_top_k=1,
                    monitor=f"{task_name}_val_loss",
                    mode="max",
                    save_weights_only=True,
                )
                trainer_param["callbacks"] = [
                    early_stop,
                    checkpoint_callback,
                ]
                trainer_param["logger"] = False
                # task_params["trainer"]["logger"] = wandb_logger
                trainer = pl.Trainer(**trainer_param)
                trainer.fit(
                    model=model,
                    train_dataloaders=datamodule.train_dataloader(),
                    val_dataloaders=datamodule.val_dataloader(),
                )
                print(f"Loading best model from {checkpoint_callback.best_model_path}")
                best_model = type(model).load_from_checkpoint(
                    checkpoint_callback.best_model_path
                )
                best_model.eval()
                metric = trainer.test(
                    model=best_model,
                    dataloaders=datamodule.test_dataloader(),
                )[0]
                results[
                    f"lr:{setting[0]}-batch:{setting[1]}-drop:{setting[2]}-l2:{setting[3]}"
                ] = metric

            if os.getenv("DATASET_LOCATION", "LOCAL") == "S3":
                aws_key = os.environ.get("AWS_ACCESS_KEY_C4DM02")
                aws_secret = os.environ.get("AWS_SECRET_KEY_C4DM02")
                s3 = boto3.client(
                    "s3",
                    endpoint_url="https://ceph-private-object-rgw.comp-research.qmul.ac.uk",
                    aws_access_key_id=aws_key,
                    aws_secret_access_key=aws_secret,
                )
                s3_path = config["s3_path"]
                s3_key = f"{s3_path}/{task_name}_probing_results_size{train_size}.json"
                result_b = json.dumps(results, indent=4).encode("utf-8")
                s3.put_object(Bucket="c4dm-02", Key=s3_key, Body=result_b)
            else:
                with open(
                    Path(root_path)
                    / f"{task_name}_probing_results_size{train_size}.json",
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, default="config.yml")
    parser.add_argument("--ckpt", dest="ckpt", type=str, default=None)
    args = parser.parse_args()
    config_file = args.config
    pl.seed_everything(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    evaluate(config, args.ckpt)


if __name__ == "__main__":
    main()
