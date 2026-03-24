import unittest
import numpy as np
from Week_01.tasks.perceptron import  task07

class TestSigmoid(unittest.TestCase):

    def test_when_x_is_0_5_then_sigmoid_returns_0_6224593312018546(self):
        # Arrange
        x = 0.5
        expected = 0.6224593312018546

        # Act
        
        actual = task07.sigmoid(x)
        # Assert

        self.assertEqual(actual, expected)
