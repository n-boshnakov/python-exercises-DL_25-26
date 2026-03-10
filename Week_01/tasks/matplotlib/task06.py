import matplotlib.pyplot as plt
from task05 import LifeExpColor, life_exp_2007, gdp_cap_usd, pop, colors

class LifeExpColorChart(LifeExpColor):
    def __init__(self, gdp_cap, life_exp, popul, year, color):
        super().__init__(gdp_cap, life_exp, popul, year, color)

    def scatter_plot_improve(self):
        def double(num):
            return num*2
        plt.scatter(self.gdp_cap, self.life_exp, list(map(double,self.popul)), self.color)
        plt.xscale("log")
        plt.ylabel('Life Expectancy [in years]')
        plt.xlabel('GDP per capita [in USD]')
        plt.xticks([1000, 10000, 100000], ["1k", "10k", "100k"])
        plt.title(f"World development in {self.year}")
        plt.grid(True)
        plt.text(3600,79, "China")
        plt.text(1500,70, "India")

        plt.show()

def main():
    life_exp_2007_color_improve = LifeExpColorChart(gdp_cap_usd, life_exp_2007, pop, 2007, colors)

    life_exp_2007_color_improve.scatter_plot_improve()

if __name__ == '__main__':
    main()
    # Answer: A