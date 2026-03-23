import unittest
from Week_01.tasks.perceptron import task01, task04

class TestEpochs(unittest.TestCase):

    def test_when_learning_rate_is_0_0001_and_epochs_500_then_weight_better_than_epochs_200(self):
        # Arrange
        dataset = task01.create_dataset(10)
        w_baseline = 10
        w_actual = 10
        epochs_actual = 500
        epochs_baseline = 200
        learning_rate = 0.0001

        # Act
        baseline = task04.run_for_epochs(epochs_baseline, w_baseline, dataset, learning_rate)
        actual = task04.run_for_epochs(epochs_actual, w_actual, dataset, learning_rate)
        # Assert
        print(actual)
        print(baseline)
        self.assertGreater(actual, baseline)