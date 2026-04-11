import unittest
from Week_02.tasks.engineering import task01

class TestCreateValue(unittest.TestCase):

    def test_when_value_5_is_created_then_data_is_5(self):
        # Arrange
        value = task01.Value(5)
        expected_result = 5

        # Act
        actual_result = value.data

        # Assert
        self.assertEqual(expected_result, actual_result)