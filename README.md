# Text-Based_Cyberbullying_Identification

text
# Cyberbullying Detection System

## Project Overview
This project implements a Machine Learning based system to detect cyberbullying in text data. It combines keyword matching and advanced ML models to classify text content as cyberbullying or not. The system features a Streamlit web interface for real-time predictions, feedback collection, and model retraining. It uses a publicly available dataset for training and includes comprehensive preprocessing, feature engineering, and visualization components.

## Features
- Automatic detection of cyberbullying keywords.
- Machine learning classification using Logistic Regression, Random Forest, and SVM models.
- Balances dataset using SMOTE to handle class imbalance.
- Model training with hyperparameter tuning and evaluation.
- Real-time text input prediction via a user-friendly Streamlit app.
- User feedback mechanism to collect corrections and improve the model.
- Enhanced retraining pipeline incorporating feedback and historical corrections.
- Visualization of model performance including accuracy, precision, recall, F1-score, and confusion matrix in publication-friendly formats.

## Installation

### Requirements
- Python 3.7+
- Packages listed in `requirements.txt` (create with `pip freeze > requirements.txt` after installing below)
  - streamlit
  - scikit-learn
  - pandas
  - numpy
  - nltk
  - imblearn
  - matplotlib
  - fasttext

### Setup
1. Clone this repository:
git clone <repo-url>
cd <repo-folder>

text
2. Install dependencies:
pip install -r requirements.txt

text
3. Download necessary NLTK resources (this is handled automatically by the app, but can also be done manually):
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

text

## Dataset
The system uses a labeled JSON dataset containing text samples and annotations indicating cyberbullying presence. The dataset is preprocessed to clean text, remove stopwords, and apply stemming. The dataset file included is:
- `Dataset-for-Detection-of-Cyber-Trolls.json`

## Usage

### Running the Web App
To start the Streamlit application for real-time detection:
streamlit run app.py

text
- Enter text in the input box.
- Click "Predict" to see classification results with confidence scores.
- Provide feedback if prediction is incorrect and optionally retrain the model.

### Training and Evaluation
- The training pipeline is implemented in `cyberbullying_project.py`.
- It performs text preprocessing, feature extraction (TF-IDF vectorization), data balancing with SMOTE, and model training using grid search for hyperparameter tuning.
- Supports Logistic Regression, Random Forest, and SVM classifiers.
- Evaluation metrics including accuracy, precision, recall, and F1-score are printed.
- The best performing model is saved along with the vectorizer for inference.

### Visualization
- Model performance and detailed metric visualizations are generated with `Visualization-images.py`.
- Produces black-and-white friendly charts optimized for publication.
- Visualizes overall performance comparison, individual metric plots, confusion matrix, and recall vs precision charts.

## Feedback and Retraining
- User feedback on predictions is saved persistently in a CSV file (`feedbackdata.csv`).
- Enhanced retraining integrates all feedback and historical corrections for model improvement.
- Retraining can be triggered from the Streamlit app or from the main pipeline.

## Project Structure
├── app.py # Streamlit web application for interactive detection
├── cyberbullying_project.py # Core ML pipeline for preprocessing, training, predicting, feedback
├── Dataset-for-Detection-of-Cyber-Trolls.json # Labeled dataset for training
├── Visualization-images.py # Scripts to generate performance visualization charts
├── feedbackdata.csv # Feedback storage file (auto-generated)
└── README.md # This file

text

## Contribution
Contributions and suggestions are welcome. Please open issues or pull requests for improvements or feature requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

This project helps automate the identification of harmful cyberbullying content to foster a safer online environment through machine learning and user feedback integration.
