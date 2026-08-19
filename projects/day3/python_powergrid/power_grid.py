import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from IPython.display import display, clear_output
from dataclasses import dataclass, asdict, replace
import networkx as nx

def plot_network(A, P, seed=11):
    G = nx.from_numpy_array(A)
    pos = nx.kamada_kawai_layout(G)
    fig, (ax, axbar) = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [1.8, 1]})
    vmax = 1.2*np.max(np.abs(P)); norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    widths = 1 + 4 * weights / weights.max() if len(weights) else []
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=P, cmap='BrBG', vmin=-vmax, vmax=vmax, node_size=600, edgecolors='k', width=widths/2)
    axbar.barh(range(len(P)), P, color=plt.cm.BrBG(norm(P)), edgecolor='k')
    axbar.axvline(0, color='k', lw=1); axbar.set_yticks(range(len(P)))
    axbar.set(xlabel='Power input', title=r"$\leftarrow$ Consumer / Producer $\rightarrow$")
    axbar.invert_yaxis()
    plt.tight_layout()
    plt.show()
    return fig, (ax, axbar)

class KuramotoPowerGrid:
    def __init__(self, N, K, adjacency_matrix, power_inputs, seed=11):
        self.N = N
        self.K = K
        self.A = adjacency_matrix
        self.P = power_inputs - np.mean(power_inputs)

        if np.sum(power_inputs) != 0.0:
            print(f"Warning: Power inputs did not balance to zero ({np.sum(power_inputs):2.2e}) and were shifted to ensure this.")
        if len(self.P) != self.N:
            raise(ValueError(f"The length of the power input array ({len(self.P)}) must equal the number of oscillators ({self.N})."))
        if self.A.shape != (self.N, self.N):
            raise(ValueError(f"Adjacency matrix must have shape {self.N, self.N}, not {self.A.shape}."))

        plot_network(self.A, self.P, seed=seed)

    def right_hand_side(self, thetas):
        rhs = np.zeros(self.N)
        for i in range(self.N):
            rhs[i] = self.P[i] + self.K/self.N * np.sum(self.A[:,i]*np.sin(thetas - thetas[i]))
        return rhs

    def runge_kutta(self, thetas, dt):
        k1 = self.right_hand_side(thetas)
        k2 = self.right_hand_side(thetas + k1*dt/2)
        k3 = self.right_hand_side(thetas + k2*dt/2)
        k4 = self.right_hand_side(thetas + k3*dt/2)

        return thetas + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    def order_parameter(self, thetas):
            return np.abs(np.sum(np.exp(1j*thetas)))/len(thetas)

    def simulate(self, T,
        dt = 0.01,
        init = None,
        live_plot = False,
        plot_interval = 100):

        if init is None:
            init = np.random.normal(scale=0.1, size=self.N)

        time = np.arange(0.0, T, dt)
        order_param = np.zeros(len(time))
        result = np.zeros((self.N, len(time)))
        result[:,0] = init
        order_param[0] = self.order_parameter(result[:, 0])

        if live_plot:
            fig, artists = self._init_live_plot(init)
            self._update_live_plot(fig, artists, time, result, order_param, 0)

        for t in range(1, len(time)):
            result[:,t] = self.runge_kutta(result[:,t-1], dt)
            order_param[t] = self.order_parameter(result[:, t])

            if live_plot:    
                if t % plot_interval == 0 or t == len(time) - 1:
                    self._update_live_plot(fig, artists, time, result, order_param, t)

        if live_plot:
            plt.close(fig)

        return result, time, order_param


    # Plotting stuff
    def _init_live_plot(self, init):
        vmax = 1.2 * np.max(np.abs(self.P))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        colors = plt.cm.BrBG(norm(self.P))

        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)
        ax_phase = fig.add_subplot(gs[0, 0])
        ax_order = fig.add_subplot(gs[1, 0], sharex=ax_phase)
        ax_circle = fig.add_subplot(gs[:, 1])
 
        phase_lines = [ax_phase.plot([], [], color=colors[i], lw=1)[0] for i in range(self.N)]
        ax_phase.set_ylim(-np.pi, np.pi)
        ax_phase.set(yticks=[-np.pi,0,np.pi], yticklabels=[r"$-\pi$", "0", r"$\pi$"])
        ax_phase.set_ylabel(r"$\theta_i$")
        ax_phase.set_title("Phase angles")
        ax_phase.axhline(0, color="grey", lw=0.5, zorder=0)
        plt.setp(ax_phase.get_xticklabels(), visible=False)
 
        order_line, = ax_order.plot([], [], color="black", lw=1.5)
        ax_order.set_ylim(0, 1.05)
        ax_order.set_xlabel("time")
        ax_order.set_ylabel("r")
        ax_order.set_title("Order parameter")
 
        circle_theta = np.linspace(0, 2*np.pi, 200)
        ax_circle.plot(np.sin(circle_theta), np.cos(circle_theta), color="grey", lw=1, zorder=0)
        ax_circle.axhline(0, color="grey", lw=0.5, zorder=0)
        ax_circle.axvline(0, color="grey", lw=0.5, zorder=0)
        scatter = ax_circle.scatter(np.sin(init), np.cos(init), c=colors, s=60,
                                     edgecolors="black", linewidths=0.5, zorder=3)
        order_arrow, = ax_circle.plot([0, 0], [0, 0], color="black", lw=2, zorder=2)
        ax_circle.set_xlim(-1.2, 1.2)
        ax_circle.set_ylim(-1.2, 1.2)
        ax_circle.set_aspect("equal")
        ax_circle.set_title("Oscillators on unit circle")
        ax_circle.axis("off")
 
        fig.tight_layout()
 
        return fig, dict(phase_lines=phase_lines, order_line=order_line,
                          scatter=scatter, order_arrow=order_arrow)

    def _update_live_plot(self, fig, artists, time, result, order_param, t):
        """Refresh all artists with data up to index t and redraw."""
        wrapped = np.angle(np.exp(1j * result[:, :t+1]))  # wrap phases to (-pi, pi]
 
        for i, line in enumerate(artists["phase_lines"]):
            line.set_data(time[:t+1], wrapped[i])
 
        artists["order_line"].set_data(time[:t+1], order_param[:t+1])

        ax_phase = artists["phase_lines"][0].axes
        ax_phase.set_xlim(0, max(time[t], 1e-6))
 
        current = result[:, t]
        artists["scatter"].set_offsets(np.column_stack([np.sin(current), np.cos(current)]))
 
        z = np.mean(np.exp(1j * current))  # complex order parameter r*e^{i*psi}
        artists["order_arrow"].set_data([0, np.imag(z)], [0, np.real(z)])
 
        clear_output(wait=True)
        display(fig)

    
    




