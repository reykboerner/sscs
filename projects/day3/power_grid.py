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

    def simulate(self, T,
        dt = 0.01,
        init = None):

        if init is None:
            init = np.random.normal(scale=0.1, size=self.N)

        time = np.arange(0.0, T, dt)
        result = np.zeros((self.N, len(time)))
        result[:,0] = init

        for t in range(1, len(time)):
            result[:,t] = self.runge_kutta(result[:,t-1], dt)

        return result, time

    def init_plot(self):
        fig, ax = 

    def order_parameter(self, thetas):
        
        return np.abs(np.sum(np.exp(1j*x)))/len(x)

    

def thetadot(theta, omega, K, A):
# The right-hand side of the kuramota ODE
    N = len(theta)
    dthetadt = np.zeros(N)
    
    #for i in range(N):
    #    dthetadt[i] += omega[i]
    #    for j in range(N):
    #        if A[i,j] == 1:
    #            dthetadt[i] += K / N * np.sin(theta[j]-theta[i])
    
    for i in range(N):
        dthetadt[i] = omega[i] + K / N * np.sum( A[:,i] * np.sin( theta - theta[i] ) )
    
    return dthetadt

def rungekutta(thetaOld, omega, K, A, dt):
# Perform runge Kutta 4 method to do the ODE evolution
    k1 = thetadot(thetaOld, omega, K, A)
    k2 = thetadot(thetaOld + dt/2 * k1, omega, K, A)
    k3 = thetadot(thetaOld + dt/2 * k2, omega, K, A)
    k4 = thetadot(thetaOld + dt/2 * k3, omega, K, A)
    
    thetaNew = thetaOld + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return thetaNew




def Kuramoto_simulator(par,sim_set):
    # THE MAIN FUNCTION THAT PERFORMS THE SIMULATION AND CALLS THE RIGHT FUNCTIONS FOR TIMESTEP AND VISUALISATIONS
    
    ## Obtain parameters
    N = par.N
    K = par.K
    
    A = par.A
    orderParameter = par.orderParameter
    
    updatePlot = sim_set.updatePlot
    plotEvolutions = sim_set.plotEvolution
    
    ## Initialize plot
    if updatePlot:
        plt.ion()
        fig_graph, ax_graph = plt.subplots(1,1,figsize = (7.5,7.5))
        G = nx.from_numpy_array(A)
        poss = nx.spring_layout(G)
        graph_plot = nx.draw_networkx(G, pos = poss, ax = ax_graph, node_size = 50, width = 0.5, with_labels = False, alpha = 0.5)
        xPositions = []
        yPositions = []
        for P in poss.values():
            xPositions.append(P[0])
            yPositions.append(P[1])
        metronomePlot, = ax_graph.plot(xPositions, yPositions, marker = '*', c = 'r', ls = '')
        
        
    
    ## Solve the kuramoto differential equation
    T = sim_set.T
    dt = sim_set.dt
    times = np.arange(0,T,dt) # in MATLAB 0:dt:T
    
    # initial conditions
    omega = np.random.rand(N) # uniform distribution between 0 and 1 for frequencies
    theta0 = np.random.normal(0,1,N) # random normal initial conditions
    theta = theta0
    
    # for saving
    thetas = np.empty((N, len(times)))
    thetas[:] = np.nan
    orderPars = np.empty(len(times))
    orderPars[:] = np.nan
    
    for j in range(len(times)):
        theta = rungekutta(theta, omega, K, A, dt)
        
        if updatePlot and j%sim_set.plot_interval == 0:
            metronomePlot.set_data(xPositions + 0.0 * np.cos(theta),yPositions + 0.1 * np.sin(theta))
            #fig_graph.canvas.draw()
            #fig_graph.canvas.flush_events()
            clear_output(wait=True)
            display(fig_graph)
        
        thetas[:,j] = theta;
        orderPars[j] = orderParameter(theta)
        
    ## Plotting after the simulation
    if plotEvolutions:
        figs, axs = plt.subplots(1,2, figsize = (10,5))
        ax_thetas = axs[0]
        ax_orderpar = axs[1]
        for k in np.linspace(0, len(thetas)-1, 20, dtype='int'):
            ax_thetas.plot(times, thetas[k,:]%(2*np.pi))
        ax_thetas.set_xlabel('t')
        ax_thetas.set_ylabel('theta_i (mod 2 pi)')
        
        ax_orderpar.plot(times, orderPars)
        ax_orderpar.set_xlabel('t')
        ax_orderpar.set_ylabel('order parameter r')
        
    return [thetas, orderPars]
        
        
    
    