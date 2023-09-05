# Music Genre Classification with Machine Learning

## Introduction

This project aims to classify music genres using various machine learning techniques. The dataset used for this project contains audio tracks from different genres, and we will explore three different approaches for genre classification: Mel-Spectrogram based classification, Convolutional Neural Network (CNN) based classification, and Long Short-Term Memory (LSTM) based classification.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `librosa`
- `matplotlib`
- `scikit-learn`
- `tensorflow` (for CNN and LSTM)
- `keras` (for CNN and LSTM)

You can install these libraries using `pip`:

```bash
pip install numpy pandas librosa matplotlib scikit-learn tensorflow keras
```

### Data Preparation

1. Download the dataset containing audio tracks from different genres and unzip it.

2. The project assumes that you have a CSV file (`features_3_sec.csv`) containing features extracted from the audio tracks. Update the `label_csv` variable in the code with the correct file path.

## Project Components

### 1. Mel-Spectrogram Based Classification

- Extracts Mel-Spectrogram features from audio tracks.
- Utilizes scikit-learn's K-Nearest Neighbors (kNN) classifier for genre classification.
- Displays confusion matrix and classification report.

### 2. Convolutional Neural Network (CNN) Based Classification

- Augments audio features for training data.
- Defines a CNN model for genre classification.
- Trains the model and displays training/validation loss and accuracy.
- Evaluates the model on the test data.
- Displays confusion matrix and AUC score.

### 3. Long Short-Term Memory (LSTM) Based Classification

- Uses LSTM to model sequential audio data.
- Defines an LSTM model for genre classification.
- Trains the model and displays training/validation loss and accuracy.
- Evaluates the model on the test data.
- Displays confusion matrix and AUC score.

## Running the Code

- Follow the code comments and instructions within each section to run the specific part of the project you're interested in.

## Results

- Each classification approach provides different insights into music genre classification. The Mel-Spectrogram-based approach may be simple but effective, while CNN and LSTM models offer more complex and powerful classification capabilities.

## Author

Roshaan Abbas Jaffery

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

Replace `[Your Name]` with your name or the name of the project author. You can also add additional sections or details as needed.
