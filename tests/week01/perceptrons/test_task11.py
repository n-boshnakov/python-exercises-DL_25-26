import unittest
import numpy as np
import torch
import torch.nn as nn
from Week_01.tasks.perceptron import task06, task08, task11

class TestSquareModel(unittest.TestCase):

    def test_when_initialized_without_specifications_then_hidden_layers_set_to_1(self):
        # Arrange
        dataset_initial = []
        for n in range(101):
            curr_set = [n, n**2]
            dataset_initial.append(curr_set)
        dataset = torch.tensor(dataset_initial)

        # dataset = [(1,1), (2,4), (3, 9), (4, 16), (5, 25), (6, 36), (7, 49), (8, 64), (9, 81), (10, 100)]
        expected_hidden_size = 1

        modelSquare = task11.ModelSquare(dataset, 1, 1, 1)

        # Act
        actual_hidden_size = len(modelSquare.hidden_layers)

        # Assert
        self.assertEqual(expected_hidden_size, actual_hidden_size)


    def test_when_train_then_all_weights_change(self):
        # Arrange
        dataset = []
        for n in range(1001):
            curr_set = [n]
            dataset.append(curr_set)
            
        hidden_size = 2
        input_size = 1

        modelSquare = task11.ModelSquare(dataset, input_size, hidden_size)
        initial_input_weights = modelSquare.weights_input
        initial_hidden_weights = modelSquare.weights_hidden

        # Act
        modelSquare.train(dataset[0][0])
        updated_input_weights = modelSquare.weights_input
        updated_hidden_weights = modelSquare.weights_hidden

        # Assert
        self.assertNotEqual(initial_input_weights, updated_input_weights)
        self.assertNotEqual(initial_hidden_weights, updated_hidden_weights)

    def test_when_forward_then_single_output_returned(self):
        # Arrange
        dataset = []
        for n in range(1001):
            curr_set = [n]
            dataset.append(curr_set)

        hidden_size = 4
        input_size = 1

        modelSquare = task11.ModelSquare(dataset, input_size, hidden_size)

        expected_output = 1

        # Act
        actual_output = modelSquare.forward(dataset)

        # Assert
        self.assertEqual(expected_output, actual_output)

