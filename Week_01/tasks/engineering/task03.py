try:
    from . import task02, task01
except ImportError:
    import task02, task01


def empire_state_climb_fixed(curr_step):
    before_step = curr_step
    rand = task01.d6_roll()
    steps = 0
    if rand < 3:
        # fixed bug in task01
        if (curr_step != 0):
            steps -= 1
        else:
            exit
    elif rand == 6:
        steps = task01.d6_roll()
    else:
        steps += 1
    curr_step += steps
    print(
        f"Before throw step = {before_step}\nAfter throw dice = {rand}\nAfter throw step = {curr_step}"
    )
    return curr_step


def main():
    curr_step = 0
    steps_list = task02.perform_steps_times(curr_step, 100,
                                            empire_state_climb_fixed)

    print(steps_list)

if __name__ == '__main__':
    main()
