import unittest
import numpy as np
from hiddenbayes import HiddenNaiveBayes

class TestHiddenNaiveBayes(unittest.TestCase):
    def setUp(self):
        self.model = HiddenNaiveBayes(num_hidden_states=2)

    def test_fit(self):
        X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        y = np.array([0, 1, 0])
        self.model.fit(X, y)
        self.assertIsNotNone(self.model.class_probs_)
        self.assertIsNotNone(self.model.cond_probs_)

    def test_predict(self):
        X_train = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        y_train = np.array([0, 1, 0])
        self.model.fit(X_train, y_train)

        X_test = np.array([[1, 0, 0], [0, 1, 1]])
        predictions = self.model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))

if __name__ == "__main__":
    unittest.main()
