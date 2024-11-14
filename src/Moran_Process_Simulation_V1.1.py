import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def initialize_graph(n, k):
    graph = nx.gnm_random_graph(n, k)
    for node in graph.nodes():
        graph.nodes[node]['fitness'] = 1  # Initial uniform fitness
        graph.nodes[node]['state'] = 0  # Initial state
    return graph

def initialize_states(graph, fraction_infected=0.1):
    for node in graph.nodes():
        graph.nodes[node]['state'] = 1 if random.random() < fraction_infected else 0

def moran_step(graph):
    total_fitness = sum(graph.nodes[node]['fitness'] * graph.nodes[node]['state'] for node in graph.nodes())
    if total_fitness == 0:
        return graph  # No nodes to reproduce

    reproduction_node = random.choices(
        population=list(graph.nodes()),
        weights=[graph.nodes[node]['fitness'] * graph.nodes[node]['state'] for node in graph.nodes()],
        k=1
    )[0]

    neighbors = list(graph.neighbors(reproduction_node))
    if neighbors:
        replacement_node = random.choice(neighbors)
        graph.nodes[replacement_node]['state'] = graph.nodes[reproduction_node]['state']
    return graph

def visualize_graph(graph, ax, info_ax, pos):
    ax.clear()
    node_color = ['blue' if graph.nodes[node]['state'] == 0 else 'red' for node in graph.nodes()]
    nx.draw(graph, pos=pos, ax=ax, node_color=node_color, with_labels=True)

    # Update the information text
    infected_count = sum(1 for node in graph.nodes() if graph.nodes[node]['state'] == 1)
    average_fitness = sum(graph.nodes[node]['fitness'] for node in graph.nodes()) / len(graph.nodes())
    total_nodes = len(graph.nodes())
    total_edges = len(graph.edges())
    info_text = f"Nodes: {total_nodes}, Edges: {total_edges}\nInfected: {infected_count}\nAverage Fitness: {average_fitness:.2f}"
    info_ax.clear()
    info_ax.axis('off')
    info_ax.text(0.02, 0.5, info_text, transform=info_ax.transAxes, ha='left', va='center')

    plt.draw()

def step(event):
    global current_graph
    current_graph = moran_step(current_graph)
    visualize_graph(current_graph, ax, info_ax, pos)

def change_fitness(delta):
    global current_graph
    for node in current_graph.nodes():
        new_fitness = current_graph.nodes[node]['fitness'] + delta
        current_graph.nodes[node]['fitness'] = max(0.1, new_fitness)  # Ensure fitness does not go below 0.1
    visualize_graph(current_graph, ax, info_ax, pos)

def increase_fitness(event):
    change_fitness(0.1)

def decrease_fitness(event):
    change_fitness(-0.1)

# Set up the figure and axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.2, bottom=0.3)
info_ax = fig.add_axes([0.01, 0.01, 0.3, 0.2])  # Adjusted to be more in the bottom left
info_ax.axis('off')  # Turn off axis for info box

button_ax1 = plt.axes([0.81, 0.05, 0.1, 0.075])
button_ax2 = plt.axes([0.68, 0.05, 0.1, 0.075])
button_ax3 = plt.axes([0.55, 0.05, 0.1, 0.075])

button_step = Button(button_ax1, 'Step')
button_inc = Button(button_ax2, 'Increase Fitness')
button_dec = Button(button_ax3, 'Decrease Fitness')

# Initialize graph and states
n = 50
k = 100
current_graph = initialize_graph(n, k)
initialize_states(current_graph, fraction_infected=0.2)
pos = nx.spring_layout(current_graph)  # Precompute positions

# Visualize the initial state and link the buttons
visualize_graph(current_graph, ax, info_ax, pos)
button_step.on_clicked(step)
button_inc.on_clicked(increase_fitness)
button_dec.on_clicked(decrease_fitness)

plt.show()