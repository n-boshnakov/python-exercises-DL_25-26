import unittest
import numpy as np
from Week_01.tasks.perceptron import task01, task02, task03

class TestDerivativeFunction(unittest.TestCase):

    def test_when_loss_from_task02_and_seed_set_to_42_then_derivative_is_105_316942235(self):
        # Arrange
        rng = np.random.default_rng(42)
        dataset = task01.create_dataset(6)
        w = rng.uniform(0, 10)
        eps = 0.01

        # Expected values calculated:
        # MSE loss(w + eps) = 303.02658629016327
        # MSE loss(w) = 301.9734168678107
        expected = 105.316942235

        # Act
        actual = round(task03.calculate_derivative(w, dataset, eps), 9)

        # Assert
        self.assertEqual(actual, expected)
    
    def test_when_running_learn_function_then_weight_reduces(self):
        # Arrange
        rng = np.random.default_rng(42)
        dataset = task01.create_dataset(6)
        w = rng.uniform(0, 10)
        eps = 0.01
        previous = w

        # Act
        actual = task03.single_step(w, dataset, eps)
        # Assert
        
        self.assertLess(actual, previous)