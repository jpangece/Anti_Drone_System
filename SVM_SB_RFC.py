# Original code was written on Google Colab 
!pip install transformers -q
!pip install datasets -q
!pip install accelerate -U -q

from transformers import set_seed
import torch
import numpy as np
seed = 42
set_seed(seed)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(pred):
  preds, labels = pred
  precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
      'f1': f1,
      'precision': precision,
      'recall': recall,
  }

"""##RCS"""

def extract_features(image):
  from scipy import stats
  image = np.array(image).flatten()
  peak_value = np.max(image)
  minimum = np.sqrt(np.sum(np.square(image)) / image.size)
  mean = np.mean(image)
  std = np.std(image)
  variance = np.var(image)
  median = np.median(image)
  mode = stats.mode(image)[0]
  features = [peak_value, minimum, mean, std, variance, median, mode]
  return features

def just_flatten_lol(image):
  add_noise = lambda x: x + np.random.normal(0,np.sqrt(0.1),x.shape)
  image = np.array(image.resize((191,191))).flatten()
  return image

from datasets import load_dataset
full_dataset = load_dataset("Goorm-AI-04/RCS_Image_Stratified_Train_Test")
test_dataset = full_dataset["test"]

from sklearn.model_selection import train_test_split
train_dataset, eval_dataset = train_test_split(
    full_dataset["train"],
    test_size=0.1,
    stratify=full_dataset["train"]["drone_type"]
)
drone_set = set(train_dataset["drone_type"])
id2label = {id:label for id, label in enumerate(drone_set)}
label2id = {label:id for id, label in id2label.items()}

train_dataset.keys()

size = [image.size for image in train_dataset["rcs_image"]]
max(size)

"""##Feature Extraction Method"""

from sklearn import svm
X = [extract_features(image) for image in train_dataset["rcs_image"]]
Y = train_dataset["label"]
clf = svm.LinearSVC()
clf.fit(X, Y)

X_val = [extract_features(image) for image in eval_dataset["rcs_image"]]
Y_val = np.array(eval_dataset["label"])
predictions = np.array(clf.predict(X_val))
compute_metrics((predictions,Y_val))

"""##Flatten Method"""

from sklearn import svm
import sklearn
X = np.array([just_flatten_lol(image) for image in train_dataset["rcs_image"]])
Y = np.array(train_dataset["label"])
lsvc = svm.LinearSVC()
svc = svm.SVC(decision_function_shape='ovo')
sgdc = sklearn.linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
models = {"lsvc":lsvc, "svc":svc, "sgdc":sgdc}
for name, model in models.items():
  print(f"training {name}")
  model.fit(X,Y)

X_val = [just_flatten_lol(image) for image in eval_dataset["rcs_image"]]
Y_val = np.array(eval_dataset["label"])
results = {}
for name, model in models.items():
  predictions = np.array(model.predict(X_val))
  results[name] = compute_metrics((predictions,Y_val))
print(results)

predictions = np.array(clf.predict(X))
compute_metrics((predictions,Y))

X_test = [just_flatten_lol(image) for image in test_dataset["rcs_image"]]
Y_test = np.array(test_dataset["label"])
test_results = {}
for name, model in models.items():
  predictions = np.array(model.predict(X_test))
  test_results[name] = compute_metrics((predictions,Y_test))
print(test_results)

"""##FMCW"""

from datasets import load_dataset
full_dataset = load_dataset("Goorm-AI-04/Drone_Doppler")

def flatten(example):
  example["image"] = np.array(example["image"]).flatten()
  return example

from sklearn.model_selection import train_test_split
train_dataset, eval_dataset = train_test_split(
    full_dataset["train"].map(flatten, num_proc=4),
    test_size=0.1,
    stratify=full_dataset["train"]["label"],
    random_state=seed
)

from datasets import Dataset
train_dataset = Dataset.from_dict(train_dataset)
eval_dataset = Dataset.from_dict(eval_dataset)
test_dataset = full_dataset["test"].map(flatten, num_proc=4)
class_set = set(train_dataset["type"])
id2label = {id:label for id, label in enumerate(class_set)}
label2id = {label:id for id, label in id2label.items()}

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier, BayesianRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.cross_decomposition import PLSCanonical, PLSSVD
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

X = np.array([example["image"] for example in train_dataset])
Y = np.array(train_dataset["label"])

lsvc = LinearSVC(dual=False)
svc = SVC(decision_function_shape='ovo')

hgbc = HistGradientBoostingClassifier()
rfc = RandomForestClassifier()

models = {"LinearSVC":lsvc,
          "SVC":svc,
          "HistGradientBoostingClassifier":hgbc,
          "RandomForestClassifier":rfc,
          }

from time import perf_counter
for name, model in models.items():
  print(f"training {name}")
  start = perf_counter()
  model.fit(X,Y)
  end = perf_counter()
  print(f"training {name} took {end-start} seconds")

import timeit
X_val = [example["image"] for example in eval_dataset]
Y_val = np.array(eval_dataset["label"])
results = {}
for name, model in models.items():
  predictions = model.predict(X_val)
  predictions = np.array(predictions)
  if name == "BayesianRidge":
    predictions = predictions.round()
  results[name] = compute_metrics((predictions,Y_val))
  print(f"results of {name}: {results[name]} / time for inference: {end-start}")

best_models = {}
best_models["LinearSVC"] = models["LinearSVC"]
best_models["HistGradientBoostingClassifier"] = models["HistGradientBoostingClassifier"]
best_models["RandomForestClassifier"] = models["RandomForestClassifier"]

X_test = [example["image"] for example in test_dataset]
Y_test = np.array(test_dataset["label"])
test_results = {}
number_of_reps = 100
for name, model in models.items():
  inference_time = []
  for i in range(number_of_reps):
    start = perf_counter()
    predictions = model.predict(X_test)
    end = perf_counter()
    inference_time.append(end-start)
  predictions = np.array(predictions)
  test_results[name] = {
    "accuracy": compute_metrics((predictions, Y_test))["accuracy"],
    "mean": np.mean(inference_time),
    "std": np.std(inference_time)
  }
  print(f"results of {name}: {test_results[name]}")

def flatten(example):
  keys = (
      "image",
      "noise_var_0.0001",
      "noise_var_0.0005",
      "noise_var_0.001",
      "noise_var_0.005",
      "noise_var_0.01"
  )
  for key in keys:
    example[key] = np.array(example[key])[:,:,0].flatten()
  return example

noise_dataset = load_dataset("Goorm-AI-04/Drone_Doppler_Noise")["train"].map(flatten, num_proc=4)
Y_test = np.array(noise_dataset["label"])
keys = (
    "image",
    "noise_var_0.0001",
    "noise_var_0.0005",
    "noise_var_0.001",
    "noise_var_0.005",
    "noise_var_0.01"
)
test_results = {}
for name, model in models.items():
  test_results[name] = {}
  for key in keys:
    X_test = [example[key] for example in noise_dataset]
    start = perf_counter()
    predictions = model.predict(X_test)
    end = perf_counter()
    inference_time = end-start
    predictions = np.array(predictions)
    test_results[name][key] = {
        "accuracy": compute_metrics((predictions, Y_test))["accuracy"],
        "time": f"{inference_time / 3497:1.3}"
    }
  print(f"results of {name}:")
  import json
  print(json.dumps(test_results[name],indent=4))
