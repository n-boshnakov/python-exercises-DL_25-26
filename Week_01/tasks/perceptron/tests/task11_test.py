from Week_01.tasks.perceptron import task11
import unittest

class TestTask11(unittest.TestCase):
    def test_when_initialized_then_weights_between_0_and_1(self):
        multi_layer_perceptron = task11.MultiLayerPerceptron(1, 10, 10, 1)
        w1 = multi_layer_perceptron.w1
        print(w1)
        print(multi_layer_perceptron.w2)
        self.assertTrue((0 < multi_layer_perceptron.w1).all() and (multi_layer_perceptron.w1 < 1).all())
        self.assertTrue((0 < multi_layer_perceptron.w2).all() and (multi_layer_perceptron.w2 < 1).all())

# return self.assertEqual(1, 1)
# return self.assertTrue(1 != 0)

if __name__ == "__main__":
    unittest.main()