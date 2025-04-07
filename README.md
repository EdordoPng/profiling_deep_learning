# profiling_deep_learning
This project is an exemple of how to apply Deep Learning techniques that aims to extract and explain leakage from a device (Profiling).

If we want to do the Profiling of a device, I suggest to apply these methods like they are ordered.

1) Classical Templating
2) Multi Layer Perceptron Neural Network (if you have nice data)
3) Convolutional Neural Networkif you suspect that your data is misaligned and cannot improve with filtering os some other method


# Side-Channel Attack on ASCAD v1 Dataset using MLP and CNN Profiling

This repository provides a complete implementation of a profiling side-channel attack (SCA) against the [ASCAD v1 dataset](https://github.com/ANSSI-FR/ASCAD), leveraging machine learning and deep learning methods, specifically Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN).

The project demonstrates how to:
- Load and preprocess ASCAD traces (fixed key scenario).
- Train profiling models on side-channel leakage traces.
- Evaluate attack effectiveness across different key bytes.
- Perform attacks using trained models to recover AES key bytes.


## 🧠 Notes on Dataset and Attack Strategy

This project uses the third S-box output during the first AES round, which has been pre-aligned to 700 time samples (range [45400..46100]).

Different desynchronization levels (desync50, desync100) can be used to simulate more realistic conditions.

Performance depends heavily on chosen model type and hyperparameters.

## 📄 Detailed Report

All technical explanations, experiments, hyperparameter choices, and results are documented in the REPORT.pdf file included in this repository.

It includes:

Background on Profiling and leakage modeling

Overview of ASCAD dataset structure

Training challenges and setup

Attack results across all AES key bytes

### 👉 We highly recommend reading the full REPORT.pdf for a complete understanding of the attack implementation and methodology.



