import unittest
from Week_02.tasks.engineering import task05

class TestValueOperation(unittest.TestCase):

    def test_when_last_operation_is_mul_then_op_returns_star(self):
        # Arrange
        value1 = task05.Value(5)
        value2 = task05.Value(7)
        expected_result = '*'

        # Act
        value1 = (value1 + value2) * value1
        actual_result = value1._op

        # Assert
        self.assertEqual(expected_result, actual_result)