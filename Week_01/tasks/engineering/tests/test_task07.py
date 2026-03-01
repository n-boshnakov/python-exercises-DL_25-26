from Week_01.tasks.engineering import task05, task07, task01
import unittest

class TestTask07(unittest.TestCase):

    def test_when_fall_chance_then_falls(self):
        return self.assertTrue(task05.multiple_walks(53, 1, task07.climb_fall_chance)[0][52] == 0)

if __name__ == "__main__":
    unittest.main()