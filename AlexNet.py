!pip install transformers -q
!pip install datasets -q
!pip install accelerate -U -q
!pip install wandb -q
!huggingface-cli whoami
!wandb login
!huggingface-cli login

from transformers import set_seed
import torch
import numpy as np
import wandb

seed = 42
set_seed(seed)

from datasets import load_dataset
full_dataset = load_dataset("Goorm-AI-04/Drone_Doppler")
test_dataset = full_dataset["test"]

from sklearn.model_selection import train_test_split
train_dataset, eval_dataset = train_test_split(full_dataset["train"], test_size=0.1, stratify=full_dataset["train"]["label"])

from datasets import Dataset
train_dataset = Dataset.from_dict(train_dataset)
eval_dataset = Dataset.from_dict(eval_dataset)

class_set = set(train_dataset["type"])
id2label = {id:label for id, label in enumerate(class_set)}
label2id = {label:id for id, label in id2label.items()}

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

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

def collate_fn(examples):
  from torchvision import transforms
  from PIL import Image
  normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  concat = lambda x : np.concatenate([x,x,x], axis=2)
  pixel_values = torch.tensor
  (
    np.array([
        np.array(
            normalize(
                concat(
                    np.expand_dims(
                        np.array(
                            Image.fromarray(
                                np.array(example["image"])
                            ).resize((224, 224))
                        ),
                        axis=2
                    )
                )
            )
        )
        for example in examples
    ])
  ).float()
  labels = torch.tensor([example["label"] for example in examples])
  return {"x": pixel_values, "labels": labels}

import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

from transformers import PreTrainedModel, PretrainedConfig

class GoogLeNetConfig(PretrainedConfig):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)


class GoogLeNet(PreTrainedModel):
  def __init__(self, model, config):
    super().__init__(config)
    self.model = model
    self.cross_entropy = torch.nn.CrossEntropyLoss()

  def forward(self, x, labels):
    logits = self.model(x)
    if labels is not None:
      loss = self.cross_entropy(logits, labels)
      return {"loss": loss, "logits":logits}
    return {"logits":logits}

def run(seed):
  if wandb.run is not None:
    wandb.finish()

  set_seed(seed)

  import torch
  model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

  import torch.nn as nn
  model.fc = nn.Linear(1024,3)

  config = GoogLeNetConfig()
  model = GoogLeNet(model, config)

  from transformers import TrainingArguments
  training_args = TrainingArguments(
      output_dir='./drive/MyDrive/FMCW/AlexNet/results',  # output directory
      num_train_epochs=12,              # total number of training epochs
      learning_rate=1e-3,
      per_device_train_batch_size=128,   # batch size per device during training
      per_device_eval_batch_size=20,   # batch size for evaluation
      warmup_steps=16,               # number of warmup steps for learning rate scheduler
      weight_decay=0.001,               # strength of weight decay
      logging_dir='./drive/MyDrive/FMCW/AlexNet/logs',            # directory for storing logs
      logging_steps=4,               # How often to print logs
      do_train=True,                   # Perform training
      do_eval=True,                    # Perform evaluation
      evaluation_strategy="epoch",     # evalute after eachh epoch
      gradient_accumulation_steps=1,  # total number of steps before back propagation
      fp16=True,                       # Use mixed precision
      run_name="FMCW_AlexNet",       # experiment name
      seed=seed,                           # Seed for experiment reproducibility
      remove_unused_columns=False,
      report_to="wandb",
  )

  from datetime import datetime
  wandb.init(
      project=f"FMCW_AlexNet",
      name = (
    f"{datetime.now().strftime('%b-%d %H:%M')} "
    f"lr:{training_args.learning_rate:1.0e} "
    f"batch_size:{training_args.per_device_train_batch_size} "
    f"epoch:{training_args.num_train_epochs}"
    )
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

  return trainer, model

end_trainer, end_model = run(seed)

wandb.finish()

end_trainer.predict(test_dataset).metrics
