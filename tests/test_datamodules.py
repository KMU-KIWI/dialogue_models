import train.datamodules as datamodules


def test_empathetic_dialogues_kr():
    dm = datamodules.EmpChat("t5-small", root="data")

    dm.prepare_data()
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
