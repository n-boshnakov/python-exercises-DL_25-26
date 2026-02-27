from Week_01.tasks.engineering import task01
import unittest

class TestDoSomethignWithAnInteger(unittest.TestCase):
    def test_when_dice_rolled_then_number_between_1_and_6(self):
        results = []
        for n in range(100):
            results.append(task01.d6_roll())
            
        return self.assertTrue(max(results) <=6 and min(results) >= 1)

if __name__ == "__main__":
    unittest.main()