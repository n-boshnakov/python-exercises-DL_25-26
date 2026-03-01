from Week_01.tasks.engineering import task03, task04, task05
import unittest


class TestTask05(unittest.TestCase):

    def test_when_walks_simulated_correct_amount(self):
        times = 5
        steps = 100
        all_walks = task05.steps_line_plot(steps, times)
        return self.assertTrue((len(all_walks) == times) and (len(all_walks[0]) == steps + 1))

if __name__ == "__main__":
    unittest.main()
