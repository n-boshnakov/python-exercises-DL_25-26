import unittest
import numpy as np
from Week_01.tasks.perceptron import task06, task08, task11

class TestSquareModel(unittest.TestCase):

    def test_when_initialized_without_specifications_then_hidden_layers_set_to_1(self):
        # Arrange
        dataset = [(1,1), (2,4), (3, 9), (4, 16), (5, 25), (6, 36), (7, 49), (8, 64), (9, 81), (10, 100)]
        expected_hidden_size = 1
        expected_initialized_hidden_length = 1

        modelSquare = task11.ModelSquare(dataset, 1, 3, 2)

        # Act
        actual_hidden_size = modelSquare.size_hidden
        actual_initialized_hidden_length = len(modelSquare.weights_hidden)

        # Assert
        self.assertEqual(expected_hidden_size, actual_hidden_size)
        self.assertEqual(expected_initialized_hidden_length, actual_initialized_hidden_length)


    def test_when_train_then_all_weights_change(self):
        # Arrange
        dataset = [(1,1), (2,4), (3, 9), (4, 16), (5, 25), (6, 36), (7, 49), (8, 64), (9, 81), (10, 100)]
        hidden_size = 2
        input_size = 1

        modelSquare = task11.ModelSquare(dataset, input_size, hidden_size)
        initial_input_weights = modelSquare.weights_input
        initial_hidden_weights = modelSquare.weights_hidden

        # Act
        modelSquare.train(1)
        updated_input_weights = modelSquare.weights_input
        updated_hidden_weights = modelSquare.weights_hidden

        # Assert
        self.assertNotEqual(initial_input_weights, updated_input_weights)
        self.assertNotEqual(initial_hidden_weights, updated_hidden_weights)

    def test_when_forward_then_all_single_output_returned(self):
        # Arrange
        dataset = [(1,1), (2,4), (3, 9), (4, 16), (5, 25), (6, 36), (7, 49), (8, 64), (9, 81), (10, 100)]
        hidden_size = 2
        input_size = 1

        modelSquare = task11.ModelSquare(dataset, input_size, hidden_size)

        expected_output = 1

        # Act
        
        actual_output = length(modelSquare.forward())

        # Assert
        self.assertEqual(expected_output, actual_output)

