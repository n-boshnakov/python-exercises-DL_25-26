import unittest
from Week_02.tasks.engineering import task04

class TestPrevValue(unittest.TestCase):

    def test_when_first_value_is_5_then_prev_returns_5(self):
        # Arrange
        value1 = task04.Value(5)
        value2 = task04.Value(7)
        expected_result = 5

        # Act
        value1 = value1 * value2
        actual_result = value1._prev[0]

        # Assert
        self.assertEqual(expected_result, actual_result)