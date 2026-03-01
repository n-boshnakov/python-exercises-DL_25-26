from Week_01.tasks.engineering import task01
import unittest

class TestTask01(unittest.TestCase):
    def test_when_dice_rolled_then_number_between_1_and_6(self):
        results = []
        for n in range(99):
            results.append(task01.d6_roll())
            
        return self.assertTrue(max(results) <=6 and min(results) >= 1)
    
    def test_when_climb_then_step_changes(self):
        curr_step = 50
        curr_step_first = curr_step
        curr_step = task01.empire_state_climb(curr_step)
        # print(f"{curr_step} is not equal to {curr_step_first}: {curr_step != curr_step_first}")
        return self.assertTrue(curr_step != curr_step_first)
    
    def test_when_climb_then_correct_step_added(self):
        # first 10 numbers generated with seed = 123:
        # [1, 5, 4, 1, 6, 2, 2, 2, 3, 2] -> expected steps if start is 50 = 50
        curr_step = 50
        # range should be 9 (1 lower), since the 6 consumes 1 roll without changing the number by itself, and having 10 dice rolls when testing will result in a different number
        for n in range(9):
            curr_step = task01.empire_state_climb(curr_step)
        
        return self.assertTrue(curr_step == 50)


if __name__ == "__main__":
    unittest.main()