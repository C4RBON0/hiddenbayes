import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class HiddenNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Hidden Naive Bayes Classifier.

    This model extends Naive Bayes by introducing hidden states, 
    which can capture underlying structures in the data.
    """

    def __init__(self, num_hidden_states=2):
        """
        Sets up the Hidden Naive Bayes model.

        Parameters:
        num_hidden_states (int): Number of hidden states (default is 2).
        """
        self.num_hidden_states = num_hidden_states  # Number of hidden states

        # These attributes will be set during training
        self.class_probs_ = None  # Class probabilities (P(C))
        self.cond_probs_ = None   # Conditional probabilities (P(X_i | C))
        self.hidden_weights_ = None  # Hidden state weights (w_ji), linking hidden states to classes

    def fit(self, X, y):
        """
        Trains the Hidden Naive Bayes model.

        This method calculates:
        - Class probabilities (P(C))
        - Conditional probabilities of features given a class (P(X_i | C))
        - Initializes hidden state weights

        Parameters:
        X -- Feature matrix (n_samples, n_features)
        y -- Target labels (n_samples,)
        
        Returns:
        self -- The trained model instance (for scikit-learn compatibility)
        """
        X = np.array(X)  # Convert X into a NumPy array if it's not already
        y = np.array(y)  # Convert y into a NumPy array

        # Step 1: Compute P(C), the probability of each class
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # Store class probabilities as a dictionary {class_label: probability}
        self.class_probs_ = {c: class_counts[i] / len(y) for i, c in enumerate(unique_classes)}


        # Step 2: Compute P(X_i | C), the probability of each feature given a class
        self.cond_probs_ = {}
        for c in unique_classes:
            indices = (y == c)  # Get indices where class is c
            X_c = X[indices]  # Extract features for class c
            
            # Apply Laplace smoothing to avoid zero probabilities
            feature_probs = (np.sum(X_c, axis=0) + 1) / (len(X_c) + 2)
            self.cond_probs_[c] = feature_probs  # Store probabilities

        # Step 3: Initialize hidden state weights equally
        self.hidden_weights_ = np.ones((self.num_hidden_states, len(unique_classes))) / self.num_hidden_states

        return self  # Return self for scikit-learn compatibility

    def predict(self, X):
        """
        Predicts class labels for input samples using the trained model.

        Parameters:
        X -- Feature matrix of shape (n_samples, n_features)

        Returns:
        predictions -- Array of predicted class labels (n_samples,)
        """
        X = np.array(X)  # Ensure X is a NumPy array
        predictions = []

        # Iterate over each sample in X
        for sample in X:
            class_scores = {}  # Store probability scores for each class

            # Compute probability for each class
            for c in self.class_probs_:
                prior = np.log(self.class_probs_[c])  # Use log to avoid precision issues
                likelihood = np.sum(np.log(self.cond_probs_[c]) * sample)  # Compute likelihood
                class_scores[c] = prior + likelihood  # Total probability in log-space

            # Pick the class with the highest probability
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return np.array(predictions)  # Return predictions as a NumPy array
