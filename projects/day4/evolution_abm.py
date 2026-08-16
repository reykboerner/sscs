import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, replace
from abm_utils import init_plot, update_plot

@dataclass
class ABM_Parameters:
    """
    Dataclass storing the parameters for the AgentBasedModel class
    """
    N_init: int = 500           # initial population size
    n_x: int = 30               # number of grid cells in x direction
    n_y: int = 30               # number of grid cells in y direction

    # Agent properties
    m: float = 0.05             # motility
    d: float = 0.1              # death rate
    r: float = 0.75             # reproduction rate

    # maximum change of agent properties for offspring
    m_mutate_max: float = 0.0
    d_mutate_max: float = 0.0
    r_mutate_max: float = 0.0

    # Agent property limits
    m_min: float = 0.0
    m_max: float = 0.5

    d_min: float = 0.01
    d_max: float = 1.0

    r_min: float = 0.0
    r_max: float = 1.0

    keep_leftover_food: bool = True   # if True, food keeps accumulating until eaten

    # This is needed to print the parameter settings with the show() method.
    def __post_init__(self):
        pass

    def show(self):
        print("Parameters:")
        for name, value in asdict(self).items():
            print(f"  {name}: {value}")

@dataclass
class Agent:
    """
    Dataclass storing the properties of an agent.
    """
    x: int                  # x position
    y: int                  # y position
    m: float                # motility
    d: float                # death rate
    r: float                # reproduction rate
    m_mutate_max: float     # max. motility mutation
    d_mutate_max: float     # max. death rate mutation
    r_mutate_max: float     # max. reproduction rate mutation

class AgentEvolutionModel:
    """
    Agent-based model for simulating evolution in a population on a 2D domain.

    Call signature: CellularAutomaton(parameters, evolution_rule), where:
    - parameters is an instance of the CA_Parameters class
    - evolution_rule is a function with call signature f(i, j, A, n_a, n_i, parameters)
    """
    def __init__(self, parameters, food_input):
        self.p = parameters
        self.food_input = food_input

    def distribute_food(self, current_distribution, input):
        """
        Updates the food distribution by adding `input` food items randomly
        over the domain. Returns the new food distribution.
        """
        if self.p.keep_leftover_food:
            new_distribution = current_distribution
        else:
            new_distribution = current_distribution * 0.0

        for i in range(input):
            x = np.random.randint(self.p.n_x);
            y = np.random.randint(self.p.n_y);
            new_distribution[x,y] += 1
            
        return new_distribution

    def mutate(self, agent, property):
        """
        Randomly mutates the `property` of an `agent`.
        """
        max_val = getattr(self.p, property + "_max")
        min_val = getattr(self.p, property + "_min")

        max_change = getattr(agent, property + "_mutate_max")
        change = (2*np.random.rand() - 1)*max_change
        new = getattr(agent, property) + change
        new_bounded = max(min(new, max_val), min_val)

        # update agent property
        setattr(agent, property, new_bounded)

    def reproduce(self, parent):
        """
        Creates offspring from a `parent` agent that inherits its properties with
        possible small variations (mutations).
        """
        child = replace(parent)

        self.mutate(child, "m")
        self.mutate(child, "d")
        self.mutate(child, "r")

        return child

    def count_agents_in_cells(self, agents):
        """
        Returns the number of agents located in each grid cell.
        """
        count = np.zeros((self.p.n_x, self.p.n_y))

        for i in range(len(agents)):
            x_i = int(np.ceil(agents[i].x*self.p.n_x) - 1)
            y_i = int(np.ceil(agents[i].y*self.p.n_y) - 1)
            count[x_i, y_i] += 1

        return count

    def step_agent(self, agent, positions, food):
        """
        Performs one time step on a single agent, including a possible reproduction or
        death event. Returns a list of new agents: either empty (death), the same agent
        only (no reproduction) or the agent plus its child (reproduction)).
        """
        x = int(np.ceil(agent.x*self.p.n_x) - 1)
        y = int(np.ceil(agent.y*self.p.n_y) - 1)

        if food[x, y] > 0:
            birth_probability = agent.r * min(1, food[x,y]/positions[x,y])
            if birth_probability > np.random.rand(): # reproduction
                return [agent, self.reproduce(agent)]
            else: # no reproduction
                return [agent]
        else:
            if agent.d < np.random.rand(): # no death
                return [agent]
            else: # death
                return []

    def step(self, t, agents, positions, food):
        """
        Performs one time step on the entire population. Returns the updated list of
        agents, the new spatial agent distribution, and the new spatial food distribution.
        """
        added_food = int(self.food_input(t, len(agents)))
        food_distribution = self.distribute_food(food, added_food)

        # Stop simulation if no agents are left
        if len(agents) == 0:
            return agents, positions, food_distribution

        # Step all agents forward in time (with possible reproduction or death)
        new_agents = []
        for agent in agents:
            new = self.step_agent(agent, positions, food_distribution)
            new_agents.extend(new)

        # Update food distribution: every agent eats one food item
        food_distribution = np.maximum(food_distribution - positions, 0)
        
        # Random movement of agents in domain
        for agent in new_agents:
            direction = 2*np.pi*np.random.rand()
            distance = agent.m*np.random.rand()
            agent.x = (agent.x + distance*np.cos(direction)) % 1.0
            agent.y = (agent.y + distance*np.sin(direction)) % 1.0
            ##setattr(agent, "x", (distance*np.cos(direction)) % 1.0)
            #setattr(agent, "y", (distance*np.sin(direction)) % 1.0)

        # Update agent distribution
        agent_distribution = self.count_agents_in_cells(new_agents)

        return new_agents, agent_distribution, food_distribution

    def simulate(self, timesteps, live_plot=True, plot_interval=100):
        """
        Main simulation function of the AgentEvolutionModel class.

        Simulates the model for the given number of `timesteps`.

        Keyword arguments
        -----------------
        - live_plot=True: if true, plots the current system state every `plot_interval` steps.
        - plot_interval=100: plotting interval (in time steps) for live_plot.
        - dpi=300: dots per inch (resolution) of displayed figure.
        """
        positions = np.zeros((timesteps+1, self.p.n_x, self.p.n_y))
        food = np.zeros((timesteps+1, self.p.n_x, self.p.n_y))

        initial_agents = []
        for i in range(self.p.N_init):
            initial_agents.append(Agent(
                np.random.rand(),
                np.random.rand(),
                self.p.m,
                self.p.d,
                self.p.r,
                self.p.m_mutate_max,
                self.p.d_mutate_max,
                self.p.r_mutate_max
            ))

        agents = [initial_agents]
        positions[0] = self.count_agents_in_cells(initial_agents)

        _plot = init_plot(self.p, timesteps, initial_agents, food[0])

        for t in range(timesteps):
            _agents, positions[t+1], food[t+1] = self.step(
                t+1, agents[t], positions[t], food[t])
            
            agents.append(_agents)

            if live_plot and (t % plot_interval) == 0:
                update_plot(_plot, t, self.p, _agents, positions[:t+1], food[:t+1])

            if not live_plot:
                print(f"\rTime step: {t+1:5.0f}", end="", flush=True)

        update_plot(_plot, timesteps, self.p, agents[-1], positions, food)
        plt.close(_plot['fig'])

        return agents, positions, food