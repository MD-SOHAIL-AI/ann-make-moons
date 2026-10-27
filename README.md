# ANN — Make Moons Classification

Train a small PyTorch artificial neural network to separate the two curved classes in Scikit-learn's Make Moons dataset.

[View the notebook](ann-make-moons.ipynb) · [Open in Google Colab](https://colab.research.google.com/github/MD-SOHAIL-AI/ann-make-moons/blob/main/ann-make-moons.ipynb)

## Dataset

The notebook generates 1,500 samples with `make_moons`, using noise `0.22` and random seed `42`. No dataset download is needed. The stratified split contains 1,050 training, 225 validation and 225 test samples. `StandardScaler` is fitted only on the training data.

## Method

The network has two hidden layers of 16 neurons, ReLU activations and a single output for binary classification. It trains on CPU for 120 epochs with Adam and `BCEWithLogitsLoss`, keeping the weights with the lowest validation loss.

## Run

Open the notebook in Google Colab and select **Runtime > Run all**. For a local Python environment, install the dependencies:

```bash
python -m pip install torch numpy matplotlib scikit-learn jupyter
jupyter notebook ann-make-moons.ipynb
```

Run the cells in order.

## Outputs

The saved notebook includes training and validation curves, a classification report, a confusion matrix, a decision-boundary plot and predictions for new points. Its saved test accuracy is **96.0% on 225 synthetic test samples**; this describes the included execution, and rerun results can vary.

Running the notebook also saves the model and scaler values to `workshop_outputs/ann_make_moons/moon_ann.pt`. This repository contains only the notebook and this README.
