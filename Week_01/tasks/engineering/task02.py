# errors with imports, used Claude 4.5 to resolve
try:
    from . import task01
except ImportError:
    import task01

# updated for task03
def perform_steps_times(curr_step, times, fun):
    steps_list = [curr_step]
    for n in range(times):
        curr_step = int(fun(curr_step))
        steps_list.append(curr_step)
    return steps_list


def main():
    curr_step = 0
    steps_list = perform_steps_times(curr_step, 100, task01.empire_state_climb)

    print(steps_list)

    # Do you notice anything unexpected in the output?.
    """ I couldn't notice anything too unusual when looking at the data, 
    apart from the fact that perhaps faster progress should have been achieved. 
    There should be a 2/6 chance of going down a step with each roll, 
    3/6 of going up one step and 1/6 of going up with multiple steps, so statistically, 
    at ~60 rolls we should have been at ~40 steps up (instead, we're at ~30), 
    with the next 40 steps boosting us well over 60. So it would appear that the odds 
    of rolling a number are not truly equal.
    
    Could also be that we can take a step down when at 0 steps, resulting in a negative step..."""

if __name__ == '__main__':
    main()