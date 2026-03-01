from Week_01.tasks.engineering import task02
import unittest

class TestTask02(unittest.TestCase):
    def test_when_climbing_then_correct_rolls_number(self):
        rolls = 104
        result = task02.perform_steps_times(0, rolls)
        # we need to account for the initial value being added to the list
        # -> initial step + 104 iterations = 105 total values
        return self.assertTrue(len(result) == (rolls + 1))

if __name__ == "__main__":
    unittest.main()