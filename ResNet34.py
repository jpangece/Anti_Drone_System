# Original code was written on Google Colab 
!pip install transformers
!pip install datasets

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import os
import csv

drone_rcs_dataset = load_dataset("Goorm-AI-04/Drone_RCS_Measurement")

# Check Dataset content: features and number of rows
F450_HH = drone_rcs_dataset["F450_HH"]
F450_HH

def append_dictionary(dictionary, element):
  for key, value in element.items():
    dictionary[key].append(value)

group_in_frequency = {}

for data in F450_HH:
  frequency = data.pop("f")
  if frequency not in group_in_frequency:
    group_in_frequency[frequency] = {'theta': [], 'phi': [], 'RCS': []}
  append_dictionary(group_in_frequency[frequency], data)

df = pd.DataFrame.from_dict(group_in_frequency)

"""# TRANING"""

!pip install transformers -q
!pip install datasets -q
!pip install accelerate -U -q
!pip install wandb -q
!huggingface-cli whoami
!wandb login

from transformers import set_seed
import torch
import numpy as np
import wandb

seed = 42
set_seed(seed)

from datasets import load_dataset
full_dataset = load_dataset("Goorm-AI-04/RCS_Image_Stratified_Train_Test")
test_dataset = full_dataset["test"]

from sklearn.model_selection import train_test_split
train_dataset, eval_dataset = train_test_split(
    full_dataset["train"],
    test_size=0.1,
    stratify=full_dataset["train"]["drone_type"]
)

from datasets import Dataset
train_dataset = Dataset.from_dict(train_dataset)
eval_dataset = Dataset.from_dict(eval_dataset)

drone_set = set(train_dataset["drone_type"])
id2label = {id:label for id, label in enumerate(drone_set)}
label2id = {label:id for id, label in id2label.items()}

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
      'f1': f1,
      'precision': precision,
      'recall': recall,
  }

def ceiling(array, ceiling = 1):
  array[array>ceiling] = ceiling
  return array

def floor(array, floor = 0):
  array[array<floor] = floor
  return array

from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-34")

def collate_fn(examples):
  concat = lambda x: np.concatenate([x, x, x], axis=2)
  add_noise = lambda x: x + np.random.normal(0, np.sqrt(0.1), x.shape)  # adding noise
  images = [
      floor(ceiling(add_noise(np.array(example["rcs_image"]))))
      for example in examples
  ]
  pixel_values = feature_extractor(
      [concat(np.expand_dims(image, axis=2)) for image in images],
      do_rescale=False
  )
  pixel_values = torch.tensor(np.array(pixel_values["pixel_values"]))
  labels = torch.tensor([example["label"] for example in examples])
  return {"pixel_values": pixel_values, "labels": labels}

from transformers import set_seed
import torch
import numpy as np
import wandb

seed = 42
set_seed(seed)

from datasets import load_dataset
full_dataset = load_dataset("Goorm-AI-04/RCS_Image_Stratified_Train_Test")
test_dataset = full_dataset["test"]

from sklearn.model_selection import train_test_split
train_dataset, eval_dataset = train_test_split(
    full_dataset["train"],
    test_size=0.1,
    stratify=full_dataset["train"]["drone_type"]
)

from datasets import Dataset
train_dataset = Dataset.from_dict(train_dataset)
eval_dataset = Dataset.from_dict(eval_dataset)

# Calculate the number of features for ResNet-34
num_features = 512

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def ceiling(array, ceiling = 1):
  array[array>ceiling] = ceiling
  return array

def floor(array, floor = 0):
  array[array<floor] = floor
  return array

from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-34")

def collate_fn(examples):
  concat = lambda x: np.concatenate([x, x, x], axis=2)
  add_noise = lambda x: x + np.random.normal(0, np.sqrt(0.1), x.shape)  # adding noise
  images = [
      floor(ceiling(add_noise(np.array(example["rcs_image"]))))
      for example in examples
  ]
  pixel_values = feature_extractor(
      [concat(np.expand_dims(image, axis=2)) for image in images],
      do_rescale=False
  )
  pixel_values = torch.tensor(np.array(pixel_values["pixel_values"]))
  labels = torch.tensor([example["label"] for example in examples])
  return {"pixel_values": pixel_values, "labels": labels}

def run(seed):
  if wandb.run is not None:
    wandb.finish()
    
  set_seed(seed)

  from transformers import ResNetForImageClassification, ResNetConfig
  config = ResNetConfig.from_pretrained("microsoft/resnet-34")
  config.num_labels = 16

  # Add ignore_mismatched_sizes=True to handle the mismatched sizes
  model = ResNetForImageClassification.from_pretrained(
      "microsoft/resnet-34",
      config=config,
      ignore_mismatched_sizes=True
  )
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

  import torch.nn as nn
  # Create a new classifier layer
  classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(num_features, 2048),  # Adjusted output size for flattening
      nn.ReLU(),
      nn.Linear(2048, 16)  # Use 16 for the final number of classes
  )

  # Replace the original classifier with the new one
  model.classifier = classifier

  from transformers import TrainingArguments
  training_args = TrainingArguments(
      output_dir='./drive/MyDrive/RCS/ResNet/results',
      num_train_epochs=2,
      learning_rate=1e-2,
      per_device_train_batch_size=128,
      per_device_eval_batch_size=20,
      weight_decay=0.01,
      logging_dir='./drive/MyDrive/RCS/ResNet/logs',
      logging_steps=4,
      do_train=True,
      do_eval=True,
      evaluation_strategy="epoch",
      gradient_accumulation_steps=1,
      run_name="RCS_ResNet34",
      seed=seed,
      remove_unused_columns=False,
      report_to="wandb",
  )

  from datetime import datetime
  wandb.init(
      project=f"RCS_ResNet34",
      name=(
          f"{datetime.now().strftime('%b-%d %H:%M')} "
          f"lr:{training_args.learning_rate:1.0e} "
          f"batch_size:{training_args.per_device_train_batch_size} "
          f"epoch:{training_args.num_train_epochs}"
      ),
      config=training_args
  )

  from transformers import Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=compute_metrics,
      data_collator=collate_fn
  )
  trainer.train()

  # Calculate metrics on the test dataset
  test_metrics = trainer.predict(test_dataset).metrics
  return trainer, model, test_metrics
  
end_trainer, end_model, test_metrics = run(seed)
wandb.finish()

print(test_metrics)
