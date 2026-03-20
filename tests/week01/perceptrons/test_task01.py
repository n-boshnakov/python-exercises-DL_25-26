import unittest
from Week_01.tasks.perceptron import task01

class TestCreateDataset(unittest.TestCase):

    def test_when_n_is_four_then_returns_four_samples(self):
        # Arrange
        n = 4
        expected = [(0, 0), (1, 2), (2, 4), (3, 6)]

        # Act
        actual = task01.create_dataset(n)

        # Assert
        self.assertListEqual(actual, expected)

class TestInitializeWeights(unittest.TestCase):

    def test_when_x_is_one_and_y_is_fourty_two_then_output_is_between_them(self):
        # Arrange
        x = 1
        y = 42

        # Act
        actual = task01.initialize_weights(x, y)

        # Assert
        self.assertGreaterEqual(actual, x)
        self.assertLess(actual, y)
        
    def test_when_called_then_result_is_float(self):
        # Arrange
        x = 1
        y = 42
        expected_type = float

        # Act
        actual = task01.initialize_weights(x, y)

        # Assert
        self.assertIsInstance(actual, expected_type)
