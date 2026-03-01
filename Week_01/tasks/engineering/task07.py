try:
    from . import task01, task06, task03, task05
except ImportError:
    import task01, task06, task03, task05


def climb_fall_chance(curr_step):
    fall_chance = task01.rng.random()
    if fall_chance <= 0.005:
        return 0
    else:
        return task03.empire_state_climb_fixed(curr_step)


if __name__ == '__main__':
    walks_list = task05.multiple_walks(100, 20, climb_fall_chance)
    task06.steps_line_plot_multiple(walks_list)
