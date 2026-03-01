try:
    from . import task01, task06, task03, task05, task07
except ImportError:
    import task01, task06, task03, task05, task07
import matplotlib.pyplot as plt

# there appears to be a bug with counting the steps and returning the lists

def estimate_chances(tries, fun):
    results = task05.multiple_walks(100, tries, fun)
    max_steps_list = []
    for result in results:
        max_steps_list.append(result[-1])
    print(max_steps_list)
    plt.hist(max_steps_list, bins=10)
    plt.xlabel("End step")
    plt.xticks([0, 20, 40, 60, 80, 100, 120], ["0", "20", "40", "60", "80", "100", "120"])
    plt.show()

if __name__ == '__main__':
    estimate_chances(500, task07.climb_fall_chance)