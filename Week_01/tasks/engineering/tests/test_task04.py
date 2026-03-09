from Week_01.tasks.engineering import task03, task04
import unittest

class TestTask04(unittest.TestCase):

    def test_when_line_plot_called_then_no_errors(self):
        task04.steps_line_plot(task03.empire_state_climb_fixed)
        return True

if __name__ == "__main__":
    unittest.main()
