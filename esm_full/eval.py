import json
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    EsmForSequenceClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

MAX_LABEL_LENGTH = 5093

"""
facebook/esm2_t6_8M_UR50D
facebook/esm2_t33_650M_UR50D
facebook/esm1b_t33_650M_UR50S
"""
TOKEN_PATH = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(TOKEN_PATH)


class EnzymeDataset(Dataset):
    """
    Dataset class for enzyme sequence classification.
    """
    def __init__(self, file_path):
        self.json_data = json.load(open(file_path, 'r'))
        self.sequences = [data['sequence'] for data in self.json_data.values()]
        self.labels = [data['ec'] for data in self.json_data.values()]
        self.ec_lst = pickle.load(open('/scratch0/zx22/zijie/esm_ft/esm_full/data/ec_type_70.pkl', 'rb'))

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
    """Define the collate function for dataloader"""
    sequences, labels = zip(*batch)
    sequence_token = tokenizer(sequences, max_length=256, padding=True, truncation=True, return_tensors="pt")
    labels = torch.stack(labels)
    output_dict = dict(labels=labels, **sequence_token)
    return output_dict


# Path to your saved model
model_path = f"{sys.argv[3]}/checkpoint-{sys.argv[1]}"
model = EsmForSequenceClassification.from_pretrained(model_path, num_labels=MAX_LABEL_LENGTH, problem_type="multi_label_classification")
# model.cuda()


def accuracy(predication, labels):
  correct_predictions = np.sum((predication == 1) & (labels == 1))
  total_label_1 = np.sum(labels == 1)
  accuracy = correct_predictions / total_label_1
  return accuracy

def compute_metrics(p):
    """Compute metrics for evaluation."""
    outputs, labels = p
    predication = (torch.sigmoid(torch.tensor(outputs)) > 0.5).float().cpu().detach().numpy()
    f1 = f1_score(y_true=labels, y_pred=predication, average='weighted', zero_division=0)
    recall = recall_score(y_true=labels, y_pred=predication, average='weighted', zero_division=0)
    precision = precision_score(y_true=labels, y_pred=predication, average='weighted', zero_division=0)
    # em = accuracy_score(y_true=labels, y_pred=predication, normalize=True, sample_weight=None)
    em = accuracy(predication, labels)
    hamming = hamming_loss(y_true=labels, y_pred=predication)

    return {'em': em, 'precision': precision, 'recall': recall, 'f1': f1, 'hammingloss': hamming}

import wandb
wandb.login(key='471850dc0af0748ea73eb1fbf278a9075c79f11d')
wandb.init()


training_args = TrainingArguments(
    output_dir=f"/scratch0/zx22/zijie/esm_ft/esm_full/results_70",
    do_eval=True,
    no_cuda=True,
    do_train=False,
    per_device_eval_batch_size=16,
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    # eval_steps=1,
    # load_best_model_at_end=True,
    # metric_for_best_model="f1",
    # greater_is_better=True,
    # push_to_hub=False,
    # logging_dir=None,
    # logging_first_step=False,
    # logging_steps=200,
    # save_total_limit=7,
    # no_cuda=False,
    # report_to='wandb'
    )

class WeightedTrainer(Trainer):
    def __int__(self, *args, **kwargs):
          super().__int__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs).logits
        # outputs = outputs.mean(dim=1)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        labels = inputs.pop("labels").float()
        loss_fn = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        return (loss, logits, labels)


validation_dataset = EnzymeDataset(f"/scratch0/zx22/zijie/esm_ft/esm_full/data/{sys.argv[2]}.json")
accelerator = Accelerator()
model = accelerator.prepare(model)

validation_dataset = accelerator.prepare(validation_dataset)
# Initialize Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    eval_dataset=validation_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)
trainer.evaluate()
