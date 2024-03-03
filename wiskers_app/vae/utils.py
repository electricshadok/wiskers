from wiskers.vae.commands.train import TrainCLI


def get_dataset(config_path: str, set_name: str = "train"):
    cmd = TrainCLI(config_path)
    cmd.datamodule.prepare_data()  # download the data
    cmd.datamodule.setup("fit")  # prepare the training dataloader

    if set_name == "train":
        return cmd.datamodule.train_dataset
    elif set_name == "val":
        return cmd.datamodule.val_dataset
    elif set_name == "test":
        return cmd.datamodule.test_dataset
    elif set_name == "predict":
        return cmd.datamodule.predict_dataset

    return None
