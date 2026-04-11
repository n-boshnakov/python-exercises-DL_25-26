import unittest
from Week_02.tasks.engineering import task03

class TestMultiplyValue(unittest.TestCase):

    def test_when_values_are_5_and_7_then_multiplication_is_35(self):
        # Arrange
        value1 = task03.Value(5)
        value2 = task03.Value(7)
        expected_result = 5 * 7

        # Act
        actual_result = value1 * value2

        # Assert
        self.assertEqual(expected_result, actual_result)