import helper
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pathlib import Path
import mlflow

if __name__ == "__main__":
    path_to_data = Path(".") / "data"

    mlflow.set_tracking_uri("https://mlflow.dv.team/")
    mlflow.set_experiment("/iss-docker")
    mlflow.pytorch.autolog(log_models=False)

    pl.seed_everything(1234)
    code_debugging_run = False

    datamodule = helper.ISSDataModule(
        path_to_data=path_to_data,
        batch_size=32
    )

    # dataset_info = helper.read_dataset_info(args.datasets_warehouse_path, args.dataset_name)

    model = helper.ISSDocker(backbone_name="resnext50_32x4d",
                             learning_rate=0.0001,
                             )
    if code_debugging_run:
        limit_train_batches = 0.01
        limit_val_batches = 0.05
        max_epochs = 1
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        max_epochs = 100

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.005,
        patience=20,
        verbose=True,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(gpus=1,
                         precision=16,
                         # val_check_interval=1.0
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         callbacks=[early_stop_callback, lr_monitor],
                         # auto_lr_find=True,
                         # profiler="simple",
                         max_epochs=max_epochs
                         )
    # trainer.tune(datamodule=datamodule, model=model)

    trainer.fit(datamodule=datamodule, model=model)
    trainer.save_checkpoint(f"iis-docker.pth", weights_only=True)
    # test_result = trainer.test(datamodule=datamodule, model=model)
    # predict_result = trainer.predict(datamodule=datamodule, model=model, )
    # print(predict_result)
    helper.prepare_submission(model)
