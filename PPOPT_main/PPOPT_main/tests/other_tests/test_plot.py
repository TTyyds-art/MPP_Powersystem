from src.ppopt.plot import parametric_plot, plotly_plot


def test_matplotlib_plot(factory_solution):
    parametric_plot(factory_solution, show=False)


def test_plotly_plot(factory_solution):
    plotly_plot(factory_solution, show=False)
