# Original code was written on Google Colab 
!pip install transformers -q
!pip install datasets -q
!pip install accelerate -U -q
!pip install wandb -q
!wandb login
!huggingface-cli whoami

from transformers import set_seed
import torch
import numpy as np
import wandb

seed = 42
set_seed(seed)

from datasets import load_dataset
full_dataset = load_dataset("Goorm-AI-04/Drone_Doppler")

def add_dimension(example):
  concat = lambda x : np.concatenate([x,x,x], axis=2)
  example["image"] = concat(np.expand_dims(np.array(example["image"]),axis=2))
  return example

from sklearn.model_selection import train_test_split
train_dataset, eval_dataset = train_test_split(
    full_dataset["train"].map(add_dimension, num_proc=4),
    test_size=0.1,
    stratify=full_dataset["train"]["label"],
    random_state=seed
)

from datasets import Dataset
train_dataset = Dataset.from_dict(train_dataset)
eval_dataset = Dataset.from_dict(eval_dataset)

test_dataset = full_dataset["test"].map(add_dimension, num_proc=4)

class_set = set(train_dataset["type"])
id2label = {id:label for id, label in enumerate(class_set)}
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

class Collator:
  def __init__(self, noise=None, image_size=None, processor=None, model_name=None):
    self.noise = noise
    self.image_size = image_size
    self.processor = processor
    self.model_name = model_name

  def __call__(self, examples):
    images = [np.array(example["image"]) for example in examples]
    if self.noise is not None:
      norm = lambda x: x * (1.0/x.max())
      images = [
          norm(image + np.random.normal(0, np.sqrt(self.noise), image.shape))
          for image in images
      ]

    if self.image_size is not None:
      def add_dimension(image):
        concat = lambda x : np.concatenate([x,x,x], axis=2)
        image = concat(np.expand_dims(image,axis=2))
        return image
        
      from PIL import Image
      images = [
          add_dimension(
              np.array(
                  Image.fromarray(image[:, :, 0]).resize(self.image_size)
              )
          )
          for image in images
      ]

    if self.processor is not None:
      if "resnet" in self.model_name.lower():
        images = self.processor(images, do_rescale=False)["pixel_values"]
      elif "googlenet" in self.model_name.lower():
        images = [np.array(self.processor(image)) for image in images]
      elif "mobilenet" in self.model_name.lower():
        images = self.processor(images)["pixel_values"]

    pixel_values = torch.tensor(np.array(images))
    labels = torch.tensor([example["label"] for example in examples])

    return {"pixel_values": pixel_values, "labels":labels}

from transformers import PreTrainedModel, PretrainedConfig

class GoogLeNetConfig(PretrainedConfig):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)

class GoogLeNet(PreTrainedModel):
  def __init__(self, model, config):
    super().__init__(config)
    self.model = model
    self.cross_entropy = torch.nn.CrossEntropyLoss()

  def forward(self, pixel_values, labels):
    logits = self.model(pixel_values)
    if labels is not None:
      loss = self.cross_entropy(logits, labels)
      return {"loss": loss, "logits":logits}
    return {"logits":logits}

def run(model_name, seed, noise, **kwargs):
  if wandb.run is not None:
    wandb.finish()

  set_seed(seed)

  if "resnet" in model_name.lower():
    model_name = "ResNet101"
    from transformers import ResNetForImageClassification
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")

    import torch.nn as nn
    model.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2048,3)
    )
    model.num_labels = 3
    model.id2label = id2label
    model.label2id = label2id

    from transformers import ConvNextImageProcessor
    processor = ConvNextImageProcessor.from_pretrained("microsoft/resnet-101")

    image_size = None

  elif "googlenet" in model_name.lower():
    model_name = "GoogLeNet"
    import torch
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)

    import torch.nn as nn
    model.fc = nn.Linear(1024,3)
    config = GoogLeNetConfig()
    model = GoogLeNet(model, config)

    from torchvision import transforms
    processor = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_size = (224,224)

  elif "mobilenet" in model_name.lower():
    model_name = "MobileNetV2"
    from transformers import MobileNetV2ForImageClassification
    model = MobileNetV2ForImageClassification.from_pretrained("Matthijs/mobilenet_v2_1.0_224")

    import torch.nn as nn
    model.classifier = nn.Linear(1280,16)
    model.num_labels = 16

    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("Matthijs/mobilenet_v2_1.0_224")
    image_size = (224,224)

  collator = Collator(
     processor=processor,
     model_name=model_name,
     noise=noise,
     image_size=image_size
  )

  from transformers import TrainingArguments
  training_args = TrainingArguments(
      output_dir='./drive/MyDrive/RCS/ResNet/results',
      logging_dir='./drive/MyDrive/RCS/ResNet/logs',
      logging_steps=16,
      per_device_eval_batch_size=64,  # batch size for evaluation
      do_train=True,  # Perform training
      do_eval=True,  # Perform evaluation
      evaluation_strategy="epoch",
      gradient_accumulation_steps=1,  # total number of steps before back propagation
      fp16=True,  # Use mixed precision
      run_name=f"Doppler_{model_name}",
      remove_unused_columns=False,
      report_to="wandb",
      **kwargs
  )

  from datetime import datetime
  wandb.init(
      # set the wandb project where this run will be logged
      project=f"Doppler_{model_name}",
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
      data_collator=collator
  )
  trainer.train()
  return trainer, model

model_name = "resnet"  # param {type:"string"}
seed = 42  # param {type:"integer"}
noise = None  # param
num_train_epochs = 4  # param {type:"integer"}
learning_rate = 1e-2  # param
per_device_train_batch_size = 128  # param {type:"integer"}
warmup_steps = 16  # param {type:"integer"}
weight_decay = 0.001  # param {type:"number"}

kwargs = {
    "num_train_epochs":num_train_epochs,
    "learning_rate":learning_rate,
    "per_device_train_batch_size":per_device_train_batch_size,
    "warmup_steps":warmup_steps,
    "weight_decay":weight_decay
}

end_trainer, end_model = run(model_name, seed, noise, **kwargs)

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir='./drive/MyDrive/RCS/ResNet/results',
    num_train_epochs=4,
    learning_rate=1e-2,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    warmup_steps=16,
    weight_decay=0.01,
    logging_dir='./drive/MyDrive/RCS/ResNet/logs',
    logging_steps=16,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    fp16=True,
    run_name=f"Doppler_{model_name}",
    seed=seed,
    remove_unused_columns=False,
    report_to="wandb",
)
wandb.finish()

end_trainer.predict(test_dataset).metrics

from transformers import ConvNextImageProcessor
processor = ConvNextImageProcessor.from_pretrained("microsoft/resnet-101")
test_collator = Collator(
    processor=processor,
    model_name="ResNet101",
    noise=0.0001,
    image_size=None
)

end_trainer.data_collator = test_collator
end_trainer.predict(test_dataset).metrics
