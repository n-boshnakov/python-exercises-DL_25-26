import unittest
from Week_02.tasks.engineering import task02

class TestAddValue(unittest.TestCase):

    def test_when_values_are_5_and_7_then_addition_is_12(self):
        # Arrange
        value1 = task02.Value(5)
        value2 = task02.Value(7)
        expected_result = 5 + 7

        # Act
        actual_result = value1 + value2

        # Assert
        self.assertEqual(expected_result, actual_result)