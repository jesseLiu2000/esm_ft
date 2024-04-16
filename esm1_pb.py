import json
import os
import numpy as np
import sys
import wandb
import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from accelerate import Accelerator
from datetime import datetime
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

MAX_LABEL_LENGTH = 253
MODEL_PATH="facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = EsmForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=MAX_LABEL_LENGTH, problem_type="multi_label_classification")

class EnzymeDataset(Dataset):
    """
    Dataset class for enzyme sequence classification.
    """
    def __init__(self, file_path):
        self.json_data = json.load(open(file_path, 'r'))
        self.sequences = [data['sequence'] for data in self.json_data.values()]
        self.labels = [data['ec'] for data in self.json_data.values()]
        self.ec_lst = pickle.load(open('/scratch0/zx22/zijie/esm/data/ec_type_70.pkl', 'rb'))

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


accelerator = Accelerator()
train_dataset = EnzymeDataset("/scratch0/zx22/zijie/esm/train_70.json")
# train_dataset, validation_dataset = random_split(all_dataset, [int(0.8 * len(all_dataset)), len(all_dataset)-int(0.8 * len(all_dataset))])
validation_dataset = EnzymeDataset("/scratch0/zx22/zijie/esm/data/new_cut.json")


# model.classifier = nn.Linear(in_features=320, out_features=MAX_LABEL_LENGTH)

for param in model.esm.parameters():
    param.requires_grad = False


model = accelerator.prepare(model)
train_dataset = accelerator.prepare(train_dataset)
validation_dataset = accelerator.prepare(validation_dataset)

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir=f"/scratch0/zx22/zijie/esm/results_70/esm2/8m_{timestamp}",
    num_train_epochs=50,
    learning_rate=1e-03,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    remove_unused_columns=False,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=1,
    fp16=True,
    eval_accumulation_steps=2,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    push_to_hub=False,
    logging_dir=None,
    logging_first_step=False,
    save_total_limit=100,
    no_cuda=False,
    report_to='wandb'
    )

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
    em = accuracy(predication, labels)
    hamming = hamming_loss(y_true=labels, y_pred=predication)
    return {'em': em, 'precision': precision, 'recall': recall, 'f1': f1, 'hammingloss': hamming}

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


# Initialize Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

# Train and Evaluate the model
trainer.train()
save_path = os.path.join("/scratch0/zx22/zijie/esm/results_70/esm1b", f"best_model_esm2_650M_probe/{timestamp}")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
