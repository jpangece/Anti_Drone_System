# Original code was written on Google Colab 
!pip install transformers -q
!pip install datasets -q
!pip install accelerate -U -q
!pip install wandb -q
!wandb login

from transformers import set_seed
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from datasets import load_dataset
from sklearn.impute import SimpleImputer

# Set a random seed for reproducibility
seed = 42
set_seed(seed)

# Define a function to compute classification metrics
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

# Define a function to extract features from an image
def extract_features(image):
    # Replace this with your feature extraction logic
    # Example: Extracting mean and standard deviation
    features = [np.mean(image), np.std(image)]
    return features

# Load your dataset
full_dataset = load_dataset("Goorm-AI-04/RCS_Image_Stratified_Train_Test")

# Split the dataset into training and evaluation sets
train_dataset, eval_dataset = train_test_split(
    full_dataset["train"], 
    test_size=0.1, 
    stratify=full_dataset["train"]["drone_type"]
)

# Extract features and labels from the training set
X_train = [extract_features(image) for image in train_dataset["rcs_image"]]
y_train = np.array(train_dataset["label"])

# Check if X_train is not None and not empty
if X_train is not None and len(X_train) > 0:
    # Replace NaN values with zeros using SimpleImputer
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train = imputer.fit_transform(X_train)
else:
    raise ValueError("X_train is empty or None. Check your data or extract_features function.")

# Extract features and labels from the evaluation set
X_eval = [extract_features(image) for image in eval_dataset["rcs_image"]]
y_eval = np.array(eval_dataset["label"])

# Check if X_eval is not None and not empty
if X_eval is not None and len(X_eval) > 0:
    # Replace NaN values with zeros using SimpleImputer
    X_eval = imputer.transform(X_eval)
else:
    raise ValueError("X_eval is empty or None. Check your data or extract_features function.")

# Create a Random Forest classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_classifier = RandomForestClassifier(random_state=seed)

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Train the classifier with the best hyperparameters on the training data
rf_classifier_best = RandomForestClassifier(random_state=seed, **best_params)
rf_classifier_best.fit(X_train, y_train)

# Make predictions on the evaluation set
eval_predictions = rf_classifier_best.predict(X_eval)

# Compute classification metrics on the evaluation set
eval_metrics = compute_metrics((eval_predictions, y_eval))

# Print the best hyperparameters and evaluation metrics
print("Best Hyperparameters:")
print(best_params)
print("Evaluation Metrics:")
print(eval_metrics)

