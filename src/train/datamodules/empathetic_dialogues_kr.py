import pytorch_lightning as pl

from datasets import Dataset
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from train.datasets import empathetic_dialogues_kr


class EmpChat(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        max_input_length: int = 512,
        max_output_length: int = 64,
        batch_size: int = 4,
        val_batch_size: int = 8,
        num_workers=0,
        root="data/empathetic_dialogues_kr",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess(self, examples):
        task = "감정대화: "

        context = examples["context"]
        disease = examples["disease"]
        emotion = examples["emotion"]
        dialogues = examples["dialogue"]

        inputs = []
        outputs = []

        for dialogue in dialogues:
            history = []
            for i, utterance in enumerate(dialogue):
                if i % 2 == 0:
                    history.append(f"대화: {utterance}")
                    inputs.append(task + " ".join(history))
                else:
                    history.append(f"응답: {utterance}")
                    outputs.append(utterance)

            if len(inputs) > len(outputs):
                inputs.pop()

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.hparams.max_input_length,
            padding="max_length",
            truncation=True,
        )

        # encode the summaries
        labels = self.tokenizer(
            outputs,
            max_length=self.hparams.max_output_length,
            padding="max_length",
            truncation=True,
        ).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

    def setup(self, stage: str):
        python_data = empathetic_dialogues_kr.load(self.hparams.root)

        self.train_set = Dataset.from_dict(python_data["train"], split="train")
        self.val_set = Dataset.from_dict(python_data["val"], split="val")
        self.test_set = Dataset.from_dict(python_data["test"], split="test")

        self.train_set = self.train_set.map(
            self.preprocess,
            batch_size=8,
            batched=True,
            remove_columns=self.train_set.column_names,
        )

        self.train_set.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        self.val_set = self.val_set.map(
            self.preprocess,
            batch_size=8,
            batched=True,
            remove_columns=self.val_set.column_names,
        )

        self.val_set.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        self.test_set = self.test_set.map(
            self.preprocess,
            batch_size=8,
            batched=True,
            remove_columns=self.test_set.column_names,
        )

        self.test_set.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
        )
