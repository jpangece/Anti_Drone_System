# Original code was written on Google Colab 
!pip install transformers -q
!pip install datasets -q
!pip install accelerate -U -q
!pip install wandb -q
!wandb login
!pip install pretrainedmodels

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoFeatureExtractor, 
                          ResNetForImageClassification, 
                          TrainingArguments, 
                          Trainer, 
                          set_seed)
import torch
import torch.nn as nn
import numpy as np
import wandb
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set seed for reproducibility
seed = 42
set_seed(seed)

# Load your dataset
nasnet_dataset = load_dataset("Goorm-AI-04/RCS_Image_Stratified_Train_Test")
test_dataset = nasnet_dataset["test"]

# Split dataset into training and evaluation
train_dataset, eval_dataset = train_test_split(
    full_dataset["train"], 
    test_size=0.1, 
    stratify=full_dataset["train"]["drone_type"]
)

# Convert datasets to Dataset objects
train_dataset = Dataset.from_dict(train_dataset)
eval_dataset = Dataset.from_dict(eval_dataset)

# Define labels mapping
drone_set = set(train_dataset["drone_type"])
id2label = {id: label for id, label in enumerate(drone_set)}
label2id = {label: id for id, label in id2label.items()}

# Define a function to compute metrics
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# Define a function to add ceiling to an array
def ceiling(array, ceiling=1):
    array[array > ceiling] = ceiling
    return array

# Define a function to add floor to an array
def floor(array, floor=0):
    array[array < floor] = floor
    return array

# Define a feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-101")

# Define a collate function
def collate_fn(examples):
    concat = lambda x: np.concatenate([x, x, x], axis=2)
    add_noise = lambda x: x + np.random.normal(0, np.sqrt(0.1), x.shape)
    images = [
        floor(ceiling(add_noise(np.array(example["rcs_image"]))))
        for example in examples
    ]
    pixel_values = feature_extractor([
        concat(np.expand_dims(image, axis=2)) 
        for image in images
    ], do_rescale=False)
    pixel_values = torch.tensor(
        np.array(pixel_values["pixel_values"])
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Define a training function
def run(seed):
  if wandb.run is not None:
    wandb.finish()
    
  set_seed(seed)

  # Load ResNet model
  model = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

  model.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2048, 16)
  )
  model.num_labels = 16
  model.id2label = id2label
  model.label2id = label2id

  # Define training arguments
  training_args = TrainingArguments(
      output_dir='./drive/MyDrive/RCS/ResNet/results',
      num_train_epochs=3,
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
      run_name="RCS_ResNet101",
      seed=seed,
      remove_unused_columns=False,
      report_to="wandb",
  )

  # Initialize WandB
  wandb.init(
      project=f"RCS_ResNet101",
      name=(
          f"{datetime.now().strftime('%b-%d %H:%M')} "
          f"lr:{training_args.learning_rate:1.0e} "
          f"batch_size:{training_args.per_device_train_batch_size} "
          f"epoch:{training_args.num_train_epochs}"
      ),
      config=training_args
  )
  
  # Define trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=compute_metrics,
      data_collator=collate_fn
  )

  # Train the model
  trainer.train()

  # Calculate metrics on the test dataset
  test_results = trainer.predict(test_dataset)

  # Calculate additional metrics
  test_metrics = compute_metrics(test_results)

  # Print or log the eval metrics
  print("Eval Metrics:", trainer.evaluate())

  # Print or log the test metrics
  print("Test Metrics:", test_metrics)

  return trainer, model

# Run the training and evaluation
end_trainer, end_model = run(seed)
