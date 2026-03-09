import matplotlib.pyplot as plt
try:
    from . import task03, task02, task05, task04
except ImportError:
    import task03, task02, task05, task04


# def steps_line_plot_multiple(walks_list):
#     plt_list = []
#     for time in range(len(walks_list)):
#         plt = task04.steps_line_plot(walks_list[time])
#         plt_list.append(plt)
#     return plt_list.show()

# adapted from https://stackoverflow.com/questions/40073322/plotting-list-of-lists-in-a-same-graph-in-python
def steps_line_plot_multiple(walks_list):
    x = [i for i in range(0, len(walks_list[0]))]
    y = walks_list
    plt.xlabel("Throw")
    plt.title("Random Walks")
    for i in range(len(y)):
        plt.plot(x, y[i])
    plt.show()
    return True

def main():
    walks_list = task05.multiple_walks(100, 5, task03.empire_state_climb_fixed)
    steps_line_plot_multiple(walks_list)
    
if __name__ == '__main__':
    main()
