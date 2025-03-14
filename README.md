# hiddenbayes
hiddenbayes is a Python library implementing the Hidden Naive Bayes (HNB) model, designed to extend scikit-learn. It provides a simple and efficient way to classify data using a probabilistic approach while incorporating hidden states for better representation of dependencies between features.


# Installation

You can install hiddenbayes via pip:
pip install hiddenbayes

Yet, you can install it directly from the source:
git clone https://github.com/C4RBON0/hiddenbayes.git
cd hiddenbayes
pip install .

Make sure you have 'numpy' and 'scikit-learn' installed before using hiddenbayes

# Usage example

Below is a example demonstrating how to use HiddenNaiveBayes

```python
import numpy as np
from hiddenbayes.hnb import HiddenNaiveBayes

# Sample dataset (binary features)
X_train = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
y_train = np.array([0, 1, 0, 1])

# Initialize and train the model
model = HiddenNaiveBayes(num_hidden_states=2)
model.fit(X_train, y_train)

# Sample test data
X_test = np.array([[1, 0, 0], [0, 1, 1]])
predictions = model.predict(X_test)

print("Predictions:", predictions)
```

# Development Setup

git clone https://github.com/C4RBON0/hiddenbayes.git
cd hiddenbayes
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

Run tests to ensure everything is working correctly

python -m unittest discover tests

# Contributing

Contributions are welcome, to contribute:

1- Fork the repository
2- Create a feature branch (git checkout -b feature-branch)
3- Commit your changes (git commit -m "Add new feature")
4- Push to the branch (git push origin feature-branch)
5- Open a pull request.

# License

This project is licensed under the BSD-3 License

If you use this library in research or production, please consider citing or acknowledging it in your work. Your support helps improve the project.

```bibtex
@misc{HiddenBayes,
  author = {Angel Cervera Ronda},
  title = {HiddenBayes: Hidden Naive Bayes classifier for Python},
  year = {2025},
  howpublished = {\url{https://github.com/C4RBON0/hiddenbayes}}
}
```

