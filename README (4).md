# Crime Classification and Threat Ranking Model

This project builds a deep learning model using CNNs to classify crime-related text data into categories and subcategories. The model also assigns a threat score to each crime type.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Usage](#usage)
- [Results](#results)
- [Saving and Loading the Model](#saving-and-loading-the-model)

## Introduction
This project uses a **Convolutional Neural Network (CNN)** for **multi-label classification** of crime reports. The model:
- Classifies reports into **crime categories** and **subcategories**.
- Assigns a **threat score** based on predefined rankings.
- Trains using **PyTorch**, and preprocesses text using **TensorFlow's Tokenizer**.

## Dataset
The dataset should be a CSV file (`crime_data.csv`) containing at least the following columns:
- `text`: The crime report description.
- `category`: The main category of the crime.
- `subcategory`: The specific type of crime.

Example:
```csv
text,category,subcategory
"Online scam involving fake shopping site", "Financial Crimes", "Online Shopping/E-commerce Frauds"
```

## Installation
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas torch scikit-learn tensorflow
```

## Model Architecture
The CNN model follows this structure:
1. **Embedding Layer**: Converts words into dense vector representations.
2. **Convolutional Layers**: Extracts patterns from text sequences.
3. **Max Pooling**: Reduces dimensionality and keeps important features.
4. **Fully Connected Layers**: Outputs probability distributions for category and subcategory classification.

## Training and Evaluation
1. **Preprocess the dataset**:
   - Encode categories and subcategories using `LabelEncoder`.
   - Tokenize and pad text sequences.
   - Split data into training, validation, and test sets.

2. **Train the CNN model**:
   ```python
   train_model(model, train_loader, val_loader, epochs=10)
   ```

3. **Evaluate the model**:
   ```python
   evaluate_model(model, test_loader)
   ```
   The script reports **classification accuracy** for both categories and subcategories.

## Usage
### Running the Training Script
To start training, run:
```bash
python train.py
```
(Change `train.py` to the name of your script if different.)

### Predicting Crimes and Ranking by Threat Score
Once trained, the model:
- Predicts categories and subcategories for new text reports.
- Assigns a **threat score** based on predefined severity rankings.
- Saves ranked predictions to `ranked_crime_predictions.csv`.

## Results
The model provides:
- **Crime category and subcategory classification**.
- **Threat score-based ranking** of crime reports.

Example output:
| Report | Predicted Category | Predicted Subcategory | Threat Score |
|--------|-------------------|----------------------|-------------|
| "Online job scam" | Financial Crimes | Online Job Fraud | 7 |
| "Data breach incident" | Cyber Attack | Data Breach/Theft | 9 |

## Saving and Loading the Model
### Save the trained model:
```python
torch.save(model.state_dict(), "crime_cnn_model.pth")
```

### Load the model:
```python
model.load_state_dict(torch.load("crime_cnn_model.pth"))
model.eval()
```

## License
This project is open-source and can be modified as needed.
