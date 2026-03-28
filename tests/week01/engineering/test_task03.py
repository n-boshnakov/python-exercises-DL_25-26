from Week_01.tasks.engineering import task02, task03
import unittest


class TestTask03(unittest.TestCase):

    def test_when_negative_roll_then_steps_not_below_0(self):
        min_values = []
        rolls = 10
        for n in range(10):
            min_values.append(min(task02.perform_steps_times(0, rolls, task03.empire_state_climb_fixed)))
        self.assertTrue(min(min_values) >= 0)

if __name__ == "__main__":
    unittest.main()
