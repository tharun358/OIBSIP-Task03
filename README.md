# Email Spam Detector with Machine Learning

This project uses Python and machine learning to build an email spam detector. It trains a model to recognize and classify messages as spam or not spam (ham).

## Features

- Reads and processes the classic SMS Spam Collection dataset (`spam.csv`).
- Cleans and prepares raw message text for machine learning.
- Trains a Naive Bayes model using TF-IDF features.
- Evaluates model accuracy and performance metrics.
- Provides a function to predict if a new message is spam or not.

## Getting Started

### 1. Prerequisites

- Python 3.7+
- pip

### 2. Install Required Packages

```bash
pip install pandas scikit-learn
```

### 3. Add the Dataset

Download `spam.csv` (the SMS Spam Collection) and place it in the same directory as the Python script.  
Or use your own CSV file with two columns:  
- `v1` (the label: 'ham' or 'spam')
- `v2` (the message text)

### 4. Run the Code

```bash
python spam_detector.py
```

You will see accuracy, metrics, and a sample prediction.

### 5. Predict Your Own Messages

Edit the `example` variable in `spam_detector.py` or use the `predict_email` function directly.

## How It Works

- **Data Loading:** Loads and processes the CSV dataset.
- **Cleaning:** Lowercases, strips punctuation and numbers.
- **Vectorization:** Converts text to TF-IDF vectors.
- **Model:** Uses Multinomial Naive Bayes, a standard model for text classification.
- **Evaluation:** Outputs accuracy, confusion matrix, and classification report.

## Example Output

```
Accuracy: 0.98
Confusion Matrix:
 [[966   0]
  [ 18 131]]
Classification Report:
               precision    recall  f1-score   support

           0       0.98      1.00      0.99       966
           1       1.00      0.88      0.93       149

    accuracy                           0.98      1115
   macro avg       0.99      0.94      0.96      1115
weighted avg       0.98      0.98      0.98      1115

Example message: 'WINNER!! You have won a free ticket to Bahamas. Click here to claim.'
Prediction: Spam
```

## Customization

- Try other models: Logistic Regression, SVM, etc.
- Use additional feature engineering (n-grams, more cleaning).
- Try with email data (preprocess headers, etc).

## References

- [SMS Spam Collection Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [scikit-learn documentation](https://scikit-learn.org/)

---

**Note:** This project is for educational purposes and not production-ready.
