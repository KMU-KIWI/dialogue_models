from train.datasets import empathetic_dialogues_kr
from datasets import Dataset


root = "data/empathetic_dialogues_kr/dummy"


class TestEmpatheticDialoguesKrDataset:
    python_data = empathetic_dialogues_kr.load(root)

    dataset = {}
    for split in ["train", "val", "test"]:
        dataset[split] = Dataset.from_dict(python_data[split], split=split)

    def test_single_access(self):
        for k in self.dataset:
            self.dataset[k][0]

    def test_slice_access(self):
        for k in self.dataset:
            self.dataset[k][100:200:4]

    def test_key_access(self):
        for k in self.dataset:
            for key in [
                "speaker_idx",
                "context",
                "disease",
                "emotion",
                "conv_id",
                "dialogue",
            ]:
                self.dataset[k][key][0]
