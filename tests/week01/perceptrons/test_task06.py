import unittest
import numpy as np
from Week_01.tasks.perceptron import task05, task06

class TestBias(unittest.TestCase):

    def test_when_100_epochs_complete_on_and_gate_and_weights_are_8_and_bias_is_added_then_error_decreased(self):
        # Arrange
        dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        epochs = 100
        learning_rate = 0.001

        perceptron_AND = task05.Perceptron(dataset_AND, 8, 8)
        perceptron_AND_bias = task06.PerceptronBias(dataset_AND, 8, 8)
        
        # Act
        perceptron_AND.train(epochs, learning_rate)
        expected = perceptron_AND.calc_loss()
        perceptron_AND_bias.train(epochs, learning_rate)
        actual = perceptron_AND_bias.calc_loss()
        # Assert

        self.assertGreater(expected, actual)