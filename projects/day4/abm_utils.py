import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
from IPython.display import display, clear_output

def get_agent_positions(agents):
    """
    Returns the x and y positions of a list of agents as a tuple (xx, yy),
    where xx and yy are numpy arrays of len(agents).
    """
    xx, yy = [], []
    for a in agents:
        xx.append(a.x)
        yy.append(a.y)
    return np.array(xx), np.array(yy)

def get_agent_properties(agents):
    """
    Returns numpy arrays (m, d, r) of len(agents) each for a list of agents,
    where m, d and r are the arrays containing the motility, death rate and
    reproduction rate values, respectively.
    """
    m, d, r = [], [], []
    for a in agents:
        m.append(a.m)
        d.append(a.d)
        r.append(a.r)
    return np.array(m), np.array(d), np.array(r)

def get_aggregate_timeseries(data):
    return np.sum(data, axis=(1,2))

def init_plot(par, timesteps, agents, food, scale=0.9):
    """Initializes the overview plot for the AgentEvolutionModel."""
    plt.ion()
    colors = np.ones((11, 4))
    colors[1:] = plt.cm.Greens(np.linspace(0.4, 1.0, 10))
    grass = ListedColormap(colors)
    bounds = np.arange(-0.5, 11.5, 1)
    norm = BoundaryNorm(bounds, grass.N)

    fig = plt.figure(figsize=(16*scale, 10*scale))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.8, 1], hspace=0.25)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.3,1,0.07], wspace=0.25)
    ax1 = fig.add_subplot(top[0, 0])
    ax2 = fig.add_subplot(top[0, 1])
    cax = fig.add_subplot(top[0, 2])

    bottom = outer[1].subgridspec(1, 3, wspace=0.3)
    ax3 = fig.add_subplot(bottom[0, 0])
    ax4 = fig.add_subplot(bottom[0, 1])
    ax5 = fig.add_subplot(bottom[0, 2])

    ax1.set(xlabel="Time", ylabel="Count", xlim=(0, timesteps+1), ylim=(0, 1.2*par.N_init))
    ax2.set(xlabel="x", ylabel="y", title=f"Time step: {0.0:5.0f}")

    ax3.set(xlabel="Motility $m$", ylabel="Relative ocurrence (%)", xlim=(par.m_min,par.m_max), ylim=(0,100))
    ax4.set(xlabel="Death rate $d$", ylabel="Relative ocurrence (%)", xlim=(par.d_min,par.d_max), ylim=(0,100))
    ax5.set(xlabel="Reproduction rate $r$", ylabel="Relative ocurrence (%)", xlim=(par.r_min,par.r_max), ylim=(0,100))

    _agents_ts, = ax1.plot([0.0], [par.N_init], label="Agents", c="navy")
    _food_ts, = ax1.plot([0.0], [0.0], label="Food", c="mediumseagreen")
    _food = ax2.imshow(food.transpose(),
        interpolation = 'nearest', origin = 'lower', extent = (0,1,0,1), cmap=grass, norm=norm, alpha=0.8)
    ax, ay = get_agent_positions(agents)
    _agents, = ax2.plot(ax, ay, marker='1', ls='', c='navy')
    fig.colorbar(_food, cax=cax, label="Food availability", ticks=np.arange(11))
    ax1.legend(loc='upper right')

    f = {"fig": fig}
    f["ts"] = ax1
    f["xy"] = ax2
    f["m"] = ax3
    f["d"] = ax4
    f["r"] = ax5
    f["agents_ts"] = _agents_ts
    f["food_ts"] = _food_ts
    f["agents"] = _agents
    f["food"] = _food

    return f

def update_plot(f, t, par, agents, positions, food):
    """Updates the overview plot of the AgentEvolutionModel."""
    f["agents_ts"].set_data(np.arange(t+1), get_aggregate_timeseries(positions))
    f["food_ts"].set_data(np.arange(t+1), get_aggregate_timeseries(food))

    xx, yy = get_agent_positions(agents)
    f["agents"].set_data(xx, yy)
    f["food"].set_data(food[-1].transpose())
    f["xy"].set_title(f"Time step: {t:5.0f}")

    m, d, r = get_agent_properties(agents)
    f['m'].cla
    f['d'].cla
    f['r'].cla
    f['m'].hist(m, bins=50, color="k", density=False, range=(par.m_min, par.m_max))
    f['d'].hist(d, bins=50, color="k", density=False, range=(par.d_min, par.d_max))
    f['r'].hist(r, bins=50, color="k", density=False, range=(par.r_min, par.r_max))

    clear_output(wait=True)
    display(f["fig"])