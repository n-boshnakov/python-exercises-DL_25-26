import unittest
import numpy as np
from Week_01.tasks.perceptron import task06, task08, task09

class TestNANDGate(unittest.TestCase):

    def test_when_input_is_0_and_0_then_trained_NAND_gate_returns_over_0_5(self):
        # Arrange
        NAND_dataset = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]        
        perceptron_NAND = task09.MultiLayerPerceptron()
        epochs = 1000
        learning_rate = 0.001
        expected = 0.5

        perceptron_NAND.train(epochs, learning_rate)
        # Act
        actual = perceptron_NAND.predict(0, 0)

        # Assert
        self.assertGreater(actual, expected)