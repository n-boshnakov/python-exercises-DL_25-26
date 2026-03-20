import unittest
import numpy as np
from Week_01.tasks.perceptron import task02

class TestCalculateLoss(unittest.TestCase):

    def test_when_weight_is_one_and_dataset_contains_3_tuples_then_mean_squared_error_is_14_divided_by_3(self):
        # Arrange
        w = 1
        dataset = [(1, 2), (2, 4), (3, 6)]
        expected = 14 / 3

        # Act
        actual = task02.calculate_loss(w, dataset)

        # Assert
        self.assertEqual(actual, expected)