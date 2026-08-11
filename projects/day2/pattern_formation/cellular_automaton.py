from dataclasses import dataclass, asdict
import numpy as np
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

@dataclass
class CA_Parameters:
    """
    Dataclass storing the parameters for the CellularAutomaton class
    """
    # domain size
    M: int = 50       # x-direction
    N: int = 50       # y-direction

    # initial conditions
    p: float = 0.5    # fraction of initial cells that are in the 1-state

    # scale-dependent activator-inhibitor parameters
    R_a: int = 5      # activator radius
    R_i: int = 10     # inhibitor radius
    w_a: float = 0.3  # activator strength
    w_i: float = 0.1  # inhibitor strength

    def __post_init__(self):
        pass

    def show(self):
        print("Parameters:")
        for name, value in asdict(self).items():
            print(f"  {name}: {value}")

class CellularAutomaton:
    """
    Cellular automaton simulator on a 2D grid, implemented with a customizable evolution
    rule.

    Call signature: CellularAutomaton(parameters, evolution_rule), where:
    - parameters is an instance of the CA_Parameters class
    - evolution_rule is a function with call signature f(i, j, A, n_a, n_i, parameters)
    """
    def __init__(self, parameters, evolution_rule):
        self.par = parameters
        self.rule = evolution_rule

    def ind2sub(self, id):
        """
        Computes the row and column values of the id-th entry in the matrix
        """
        x = 0
        y = 0
        while id / self.par.N >= 1:
            y += 1
            id -= self.par.N
        x = id
        return [x,y]
    
    def sub2ind(self, x, y):
        """
        Computes which entry number is the entry at location x,y in the matrix
        """
        id = y * self.par.N + x
        return id

    def simulate(self, timesteps, live_plot=True, plot_interval=1):
        """
        Runs a simulation of the cellular automaton for the number of steps specifed by
        `timesteps`.

        Keyword arguments
        -----------------
        - live_plot=True: if True, plot the results during the simulation
        - plot_interval=1: number of time steps between plotting instances

        Returns the simulation results as a numpy array of shape (T, N, M), where T is
        the number of time steps and (N, M) is the grid size.
        Active (passive) sites have value 1 (0).
        """
        # Initialisation of the grid
        A0 = (np.random.rand(self.par.N, self.par.M) < self.par.p)
        A0 = A0.astype(int)
        
        # Pre-calculation of the neighbours within certain radius
        # (this makes the simulation faster)
        # Build proximity matrices
        prox_act = []
        prox_inh = []
        for dx in np.arange(0,max(self.par.R_i, self.par.R_a), 1, dtype ='int'):
            for dy in np.arange(0,max(self.par.R_i, self.par.R_a),1, dtype = 'int'):
                dist = dx*dx + dy*dy
                if dist < self.par.R_a*self.par.R_a:
                    prox_act.append( (dx,dy) )
                    prox_act.append( (dx,-dy) )
                    prox_act.append( (-dx,dy) )
                    prox_act.append( (-dx,-dy) )
                elif dist < self.par.R_i*self.par.R_i:
                    prox_inh.append( (dx,dy) )
                    prox_inh.append( (dx,-dy) )
                    prox_inh.append( (-dx,dy) )
                    prox_inh.append( (-dx,-dy) )
        
        # Then use it to find all neighbours (negative values are fine!)
        neighbours_a = []
        neighbours_i = []
        perc_fin = 0
        
        for i in range(self.par.N*self.par.M):
            [xi, yi] = self.ind2sub(i)
            NN_a = set()
            NN_i = set()
            for pos in prox_act:
                x = ( xi + pos[0] ) % self.par.M
                y = ( yi + pos[1] ) % self.par.N
                NN_a.add( (y,x) )
            for pos in prox_inh:
                x = ( xi + pos[0] ) % self.par.M
                y = ( yi + pos[1] ) % self.par.N
                NN_i.add( (y,x) )
            # remove points in NN_i that are also in NN_a
            NN_i = NN_i - NN_a
            # Then convert to the right format (x1,x2,x3,...), (y1,y2,y3,...)
            xs_a = []
            ys_a = []
            for pos in NN_a:
                xs_a.append(pos[1])
                ys_a.append(pos[0])
            neighbours_a.append((tuple(ys_a), tuple(xs_a)))
            
            xs_i = []
            ys_i = []
            for pos in NN_i:
                xs_i.append(pos[1])
                ys_i.append(pos[0])
            neighbours_i.append((tuple(ys_i), tuple(xs_i)))

        # Initialisation of plot
        if live_plot:
            plt.ion()
            fig_domain, ax_domain = plt.subplots(1,1,figsize=(5,5),dpi=200)
            plot_domain = ax_domain.imshow(A0.transpose(),
                interpolation = 'nearest', origin = 'lower',
                extent = (0,self.par.M,0,self.par.N),
                vmin = 0, vmax = 1, cmap = 'Greys', animated = True)
            ax_domain.set(xlabel="x", ylabel="y")
            ax_domain.set_title('Time step: 0')

        # Actual simulation
        A = np.zeros((timesteps+1, self.par.N, self.par.M))
        A[0] = A0

        for t in range(timesteps):
            # Loop over cells and apply evolution rules
            for i in range(self.par.M):
                for k in range(self.par.N):
                    id = self.sub2ind(i, k)
                    A[t+1,k,i] = self.rule(i, k, np.array(A[t]),
                        neighbours_a[id], neighbours_i[id], self.par)
            
            # Update plotting
            if live_plot and (t%plot_interval == 0):
                plot_domain.set_data(A[t+1].transpose())
                ax_domain.set_title('Time step: {:3.0f}'.format(t+1))
                clear_output(wait=True)
                display(fig_domain)
            
        return A