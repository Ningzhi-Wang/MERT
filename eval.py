import argparse
import os
import yaml
import json
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
    "MTT": "/import/c4dm-05/nw003/datasets/mtt",
    "GTZAN": "/import/c4dm-05/nw003/datasets/gtzan",
    "EMO": "/import/c4dm-05/nw003/datasets/musicemo",
    "Giantsteps": "/import/c4dm-05/nw003/datasets/giantstep",
    "NSynthPitch": "/import/c4dm-datasets/nsynth",
    "NSynthTimbre": "/data/scratch/acw713/datasets/nsynth",
    "MEDLEYDB": "/data/scratch/acw713/datasets/medleydb/V1/",
    "MTGGenre": "/data/scratch/acw713/datasets/mtg/",
    "MTGInstrument": "/data/scratch/acw713/datasets/mtg/",
    "PhonationModes": "/data/scratch/acw713/datasets/phonation_modes",
}



def evaluate(config, ckpt):
    assert ckpt is not None, "Checkpoint must be provided for evaluation"
    task_names = config["task_names"]
    # task_names = ["MTT", "Giantsteps", "GTZAN", "EMO"]

    # Perform hyperparameter search as in MERT
    lrs = [1e-4, 1e-3, 1e-2]
    batch_size = [64]
    drop_outs = [0.25]
    l2s = [0, 1e-2]

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
                model = task_config
                model = task.prober(**model_param)
                early_stop = EarlyStopping(
                    monitor=f"{task_name}_val_loss",
                    patience=30,
                    mode="max",
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
    os.nice(10)
    torch.set_num_threads(1)
    evaluate(config, args.ckpt)


if __name__ == "__main__":
    main()
