import numpy as np

rng = np.random.default_rng(123)

def d6_roll():
    return rng.integers(1, 7)

def empire_state_climb(curr_step):
    before_step=curr_step
    rand = d6_roll()
    steps = 0
    if rand < 3:
        steps -= 1
    elif rand == 6:
        steps = d6_roll()
    else:
        steps += 1
    curr_step += steps
    print(f"Before throw step = {before_step}\nAfter throw dice = {rand}\nAfter throw step = {curr_step}")
    return curr_step



if __name__ == '__main__':
       
    print(f"Random float: {rng.random()}")
    print(f"Random integer 1: {d6_roll()}")
    print(f"Random integer 2: {d6_roll()}")


    curr_step = 50
    curr_step = empire_state_climb(curr_step)

    # Random float: 0.6823518632481435
    # Random integer 1: 4
    # Random integer 2: 1
    # Before throw step = 50
    # After throw dice = 6
    # After throw step = 52

