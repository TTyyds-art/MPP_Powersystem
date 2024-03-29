import time
from typing import List

import matplotlib.pyplot as plt
import numpy
import numpy as np
import plotly.graph_objects as go
from math import atan2
from matplotlib import pyplot
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from .solution import Solution
from .solver import Solver
from .utils.general_utils import make_column


def vertex_enumeration_2d(A: numpy.ndarray, b: numpy.ndarray, solver: Solver) -> List[numpy.ndarray]:
    """
    Computes the vertices of a 2D polytope from the half space representation, uses a naive O(n^2) algorithm but is
    sufficient for plotting purposes

    Generates vertices for the 2D polytope of the following structure Ax <= b

    :param solver:
    :param A: The left-hand side constraint matrix
    :param b: The right-hand side constraint matrix
    :return: List of vertices
    """

    num_constrs = A.shape[0]
    trials = [[i, j] for i in range(num_constrs) for j in range(i + 1, num_constrs)]
    res = map(lambda comb: solver.solve_lp(None, A, b, comb), trials)
    filtered_res = filter(lambda x: x is not None, res)
    return list(map(lambda x: x.sol, filtered_res))  # 加了[:2]，只取最后两个元素, 删除了. 23-8-7


def sort_clockwise(vertices: List[numpy.ndarray]) -> List[numpy.ndarray]:
    """
    Sorts the vertices in clockwise order. This is important for rendering as if they were not sorted then you would see nonsense.

    :param vertices:
    :return:
    """

    center = sum(vertices, numpy.array([0, 0])) / len(vertices)
    return sorted(vertices, key=lambda x: atan2((x[1] - center[1]), (x[0] - center[0])))


# TODO: specify dimensions to fix
def gen_vertices(solution: Solution):
    """
    Generates the vertices associated with the critical regions in the solution.

    :param solution: a multiparametric region
    :return: a list of a collection of vertices sorted counterclockwise that correspond to the specific region

    """

    solver_obj = solution.program.solver
    cr_vertices = map(lambda cr: vertex_enumeration_2d(cr.E, cr.f, solver_obj), solution.critical_regions)
    sorted_vertices = map(lambda verts: sort_clockwise(verts), cr_vertices)
    return list(sorted_vertices)


def plotly_plot(solution: Solution, save_path: str = None, show=True) -> None:
    """
    Makes a plot via the plotly library, this is good for interactive figures that you can embed into webpages and handle interactively.

    :param solution:
    :param save_path: Keyword argument, if a directory path is specified it will save a html copy and a png to that directory
    :param show: Keyword argument, if True displays the plot otherwise does not display
    :return: no return, creates a graph of the solution
    """
    fig = go.Figure()
    vertex_list = gen_vertices(solution)

    for i, region_v in enumerate(vertex_list):
        x_ = [region_v[j][0] for j in range(len(region_v))]
        y_ = [region_v[j][1] for j in range(len(region_v))]

        fig.add_trace(go.Scatter(x=x_, y=y_, fill="toself", name=f'Critical Region {i}'))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000
    )

    fig.update_layout(
        hoverlabel=dict(
            bgcolor='white'
        )
    )

    if save_path is not None:
        file_tag = str(time.time())
        fig.write_image(save_path + file_tag + ".png")
        # fig.write_html(save_path + file_tag + ".html", include_plotyjs=False, full_html=False)

    if show:
        fig.show()


def parametric_plot(solution: Solution, save_path: str = None, points=None, show=True) -> None:
    # ... your other code here ...

    vertex_list = gen_vertices(solution)
    polygon_list = [Polygon(v) for v in vertex_list]

    # 定义一个固定的颜色列表，包含10种颜色
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

    # 使用模运算确定每个多边形的颜色
    colors = [color_list[i % len(color_list)] for i in range(len(solution.critical_regions))]

    # 计算绘图的行数和列数，这只是一个简单的示例，你可以根据你的具体需求调整它
    total = len(polygon_list)
    cols = int(np.ceil(np.sqrt(total)))
    rows = int(np.ceil(total / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # 这里的figsize是一个示例，你可以根据需要进行调整

    # 将2D的axes数组转换为1D，以方便迭代
    if rows > 1 and cols > 1:
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, (polygon_, ax) in enumerate(zip(polygon_list, axes)):
        p = PatchCollection(polygon_list[:i + 1], facecolors=colors[:i + 1], alpha=.8, edgecolors='black', linewidths=1)
        # p.set_array(colors[:i + 1])
        ax.add_collection(p)
        if points is not None:
            for j in range(i+1):
                ax.scatter(points[j][0], points[j][1], color='black', s=3)

        ax.set_xlim(55, 85)
        ax.set_ylim(16, 24)

    # 如果保存图像：
    if save_path is not None:
        plt.savefig(f"{save_path}.png", dpi=1000)

    # 如果要显示图像：
    if show:
        plt.show()



def parametric_plot_1D(solution: Solution, save_path: str = None, show=True) -> None:
    """
    Makes a simple plot of a 1D parametric solution

    :param solution:
    :param save_path:
    :param show:
    :return:
    """

    # check if the solution is actually 1 dimensional
    if solution.theta_dim() != 1:
        print(f"Solution is not 1D, the dimensionality of the solution is {solution.theta_dim()}")
        return None

    # see the dimensionality of the response variable x*

    # x_dim = solution.program.num_x()

    # set up the plotting object
    _, ax = pyplot.subplots()

    # plot the critical regions w.r.t. x*
    for critical_region in solution.critical_regions:
        # get extents
        boundaries = critical_region.f / critical_region.E
        y = [critical_region.evaluate(theta=make_column(boundary)).flatten() for boundary in boundaries]
        ax.plot(boundaries, y, solid_capstyle='round')

    if save_path is not None:
        pyplot.savefig(save_path + str(time.time()) + ".png", dpi=1000)

    if show:
        pyplot.show()
        time.sleep(2)  # Wait for 2 seconds
        pyplot.close()
