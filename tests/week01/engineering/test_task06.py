from Week_01.tasks.engineering import task05, task06
import unittest

class TestTask06(unittest.TestCase):

    def test_when_line_plots_called_then_no_errors(self):
        task06.steps_line_plot_multiple(task05.multiple_walks(10, 8))
        return True

if __name__ == "__main__":
    unittest.main()
