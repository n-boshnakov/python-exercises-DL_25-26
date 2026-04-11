import unittest
from Week_02.tasks.engineering import task06

class TestTraceFunc(unittest.TestCase):

    def test_when_two_operations_and_three_values_then_nodes_list_is_six(self):
        # Arrange
        value1 = task06.Value(5)
        value2 = task06.Value(7)
        value3 = task06.Value(-3)
        expected_result = 6

        # Act
        value1 = (value1 + value2) * value1
        nodes, edges = task06.trace(value1)
        actual_result = len(nodes)

        # Assert
        self.assertEqual(expected_result, actual_result)