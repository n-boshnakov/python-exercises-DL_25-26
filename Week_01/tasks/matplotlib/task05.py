import matplotlib.pyplot as plt
from task04 import LifeExp, life_exp_2007, gdp_cap_usd, pop

colors = ['red', 'green', 'blue', 'blue', 'yellow', 'black', 'green', 'red', 'red', 'green', 'blue', 'yellow', 'green', 'blue', 'yellow', 'green', 'blue', 'blue', 'red', 'blue', 'yellow', 'blue', 'blue', 'yellow', 'red', 'yellow', 'blue', 'blue', 'blue', 'yellow', 'blue', 'green', 'yellow', 'green', 'green', 'blue', 'yellow', 'yellow', 'blue', 'yellow', 'blue', 'blue', 'blue', 'green', 'green', 'blue', 'blue', 'green', 'blue', 'green', 'yellow', 'blue', 'blue', 'yellow', 'yellow', 'red', 'green', 'green', 'red', 'red', 'red', 'red', 'green', 'red', 'green', 'yellow', 'red', 'red', 'blue', 'red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue', 'red', 'blue', 'blue', 'blue', 'yellow', 'red', 'green', 'blue', 'blue', 'red', 'blue', 'red', 'green', 'black', 'yellow', 'blue', 'blue', 'green', 'red', 'red', 'yellow', 'yellow', 'yellow', 'red', 'green', 'green', 'yellow', 'blue', 'green', 'blue', 'blue', 'red', 'blue', 'green', 'blue', 'red', 'green', 'green', 'blue', 'blue', 'green', 'red', 'blue', 'blue', 'green', 'green', 'red', 'red', 'blue', 'red', 'blue', 'yellow', 'blue', 'green', 'blue', 'green', 'yellow', 'yellow', 'yellow', 'red', 'red', 'red', 'blue', 'blue']

class LifeExpColor(LifeExp):
    def __init__(self, gdp_cap, life_exp, popul, year, color):
        super().__init__(gdp_cap, life_exp, popul, year)
        self.color = color

    def scatter_plot_color(self):
        def double(num):
            return num*2
        plt.scatter(self.gdp_cap, self.life_exp, list(map(double,self.popul)), self.color)
        plt.xscale("log")
        plt.ylabel('Life Expectancy [in years]')
        plt.xlabel('GDP per capita [in USD]')
        plt.xticks([1000, 10000, 100000], ["1k", "10k", "100k"])
        plt.title(f"World development in {self.year}")

        plt.show()

if __name__ == '__main__':
    life_exp_2007_color = LifeExpColor(gdp_cap_usd, life_exp_2007, pop, 2007, colors)

    life_exp_2007_color.scatter_plot_color()