import unittest
import numpy as np
from Week_01.tasks.perceptron import task06, task08

class TestSigmoidLearning(unittest.TestCase):

    def test_when_100_epochs_complete_on_and_gate_and_weights_are_8_and_sigmoid_is_used_then_error_decreased(self):
        # Arrange
        dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        epochs = 100
        learning_rate = 0.001

        perceptron_AND = task06.PerceptronBias(8, 8)
        perceptron_AND_sigmoid = task08.PerceptronSigmoid(8, 8)
        
        # Act
        perceptron_AND.train(epochs, dataset_AND, learning_rate)
        expected = perceptron_AND.calc_loss(dataset_AND)
        perceptron_AND_sigmoid.train(epochs, dataset_AND, learning_rate)
        actual = perceptron_AND_sigmoid.calc_loss(dataset_AND)
        # Assert

        self.assertGreater(expected, actual)
