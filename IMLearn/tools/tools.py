import sys
import matplotlib
from matplotlib import rc, rcParams
from matplotlib.style import use
import matplotlib.pyplot as plt


def progressbar(it, prefix="Progress: ", size=60):
    it = list(it)
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print('\r', f"{prefix}[{u'█' * x}{'.' * (size - x)}] {j}/{count}", end='', flush=True)

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True)


def set_nicer_ploting():
    # style = 'seaborn-darkgrid'
    # style = 'ggplot'
    style = 'default'
    use(style)
    font_size = 52 if style == "seaborn" else 18
    rc('text', usetex=True)
    rc('font', **{'family': "sans-serif", 'size': font_size}, )
    pa = {'text.latex.preamble': r'\usepackage{amsmath}',
          "axes.grid": True, 'axes.axisbelow': True}
    rcParams.update(pa)
    plt.rcParams["axes.titlesize"] = 17


def plot_confidence_interval(x, variance, average, label, variance_factor=2):
    """
    plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    """
    fig, ax = plt.subplots(1)
    ax.set_axisbelow(True)
    ax.grid()
    ax.plot(x, average, lw=2, label=label, color='black')
    ax.plot(x, average + variance_factor * variance, lw=1, color='gray', alpha=0.4)
    ax.plot(x, average - variance_factor * variance, lw=1, color='gray', alpha=0.4)
    ax.fill_between(x, average + 2 * variance, average - 2 * variance, color='gray', alpha=0.2)
    return fig, ax
