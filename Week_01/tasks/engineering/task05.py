import matplotlib.pyplot as plt
try:
    from . import task03, task02
except ImportError:
    import task03, task02


def multiple_walks(steps, times, fun):
    all_walks = []
    for n in range(times):
        all_walks.append(task02.perform_steps_times(0, steps, fun))

    return all_walks


if __name__ == '__main__':
    all_walks = multiple_walks(100, 5, task03.empire_state_climb_fixed)

    print(all_walks)
