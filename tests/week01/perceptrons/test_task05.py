import unittest
import numpy as np
from Week_01.tasks.perceptron import task05

class TestGates(unittest.TestCase):

    def test_when_100_epochs_complete_on_and_gate_then_error_decreased(self):
        # Arrange
        dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        epochs = 100
        learning_rate = 0.001

        perceptron_AND = task05.Perceptron()
        expected = perceptron_AND.calc_loss(dataset_AND)

        
        # Act
        perceptron_AND.train(epochs, dataset_AND, learning_rate)
        actual = perceptron_AND.calc_loss(dataset_AND)
        # Assert

        self.assertGreater(expected, actual)

    def test_when_100_epochs_complete_on_or_gate_then_error_decreased(self):
            # Arrange
            dataset_OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
            epochs = 100
            learning_rate = 0.001

            perceptron_OR = task05.Perceptron()
            expected = perceptron_OR.calc_loss(dataset_OR)

            
            # Act
            perceptron_OR.train(epochs, dataset_OR, learning_rate)
            actual = perceptron_OR.calc_loss(dataset_OR)
            # Assert

            self.assertGreater(expected, actual)