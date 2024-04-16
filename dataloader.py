import wandb
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from accelerate import Accelerator


class EnzymeDataset(Dataset):
    """
    Dataset class for enzyme sequence classification.
    """
    def __init__(self, file_path):
        self.json_data = json.load(open(file_path, 'r'))
        self.sequences = [data['sequence'] for data in self.json_data.values()]
        self.labels = [data['ec'] for data in self.json_data.values()]
        self.ec_lst = pickle.load(open('/scratch0/zx22/zijie/esm/data/ec_type.pkl', 'rb'))

    def __len__(self):
        return len(self.sequences)

    def _get_label(self, specific_name):
        labels = [1 if name in specific_name else 0 for name in self.ec_lst]
        return labels

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        onehot_label = self._get_label(self.labels[idx])
        return sequence, torch.tensor(onehot_label)

def collate_fn(batch):
    """Define teh collate function for dataloader"""
    sequences, labels = zip(*batch)
    sequence_token = tokenizer(sequences, max_length=256, padding=True, truncation=True, return_tensors="pt")
    labels = torch.stack(labels)
    output_dict = dict(labels=labels, **sequence_token)
    return output_dict


train_dataset = EnzymeDataset("/scratch0/zx22/zijie/esm/data/train_cut.json")
validation_dataset = EnzymeDataset("/scratch0/zx22/zijie/esm/data/new_cut.json")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

for dict in train_dataloader:
    print("1")
