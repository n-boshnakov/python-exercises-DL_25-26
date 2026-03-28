import unittest
import numpy as np
from Week_01.tasks.perceptron import task06, task08, task10

class TestXORGate(unittest.TestCase):

    def test_when_input_is_1_and_1_then_trained_XOR_gate_returns_uner_0_5(self):
        # Arrange
        XOR_dataset = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]        
        perceptron_XOR = task10.PerceptronXOR(XOR_dataset)
        epochs = 1000
        learning_rate = 0.001
        expected = 0.5

        perceptron_XOR.train(epochs, learning_rate)
        # Act
        actual = perceptron_XOR.forward(0, 0)

        # Assert
        self.assertLesser(actual, expected)