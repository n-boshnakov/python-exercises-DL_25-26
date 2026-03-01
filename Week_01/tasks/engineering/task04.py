import matplotlib.pyplot as plt
try:
    from . import task03, task02
except ImportError:
    import task03, task02


# def steps_line_plot(fun):
#     list_steps = task02.perform_steps_times(0, 100, fun)
#     list_rolls = [i for i in range(0, 101)]
#     plt.xlabel('Throw')
#     plt.title('Random walk')
#     plt.plot(list_rolls, list_steps)
#     plt.show()
#     return True

def steps_line_plot(list_steps):
    list_steps = list_steps
    list_rolls = [i for i in range(0, len(list_steps))]
    plt.xlabel('Throw')
    plt.title('Random walk')
    plt.plot(list_rolls, list_steps)
    return plt


if __name__ == '__main__':
    steps_line_plot(task02.perform_steps_times(0, 100, task03.empire_state_climb_fixed)).show()
# Even though I couldn't reach the desired line plot visualization, I believe that my line plot depicts the results from task03 faithfully, as can be seen in the beginning, where the steps taken are 0, 0, 1. In the desired visualization it appears that there is an immediate jump to a step above 1