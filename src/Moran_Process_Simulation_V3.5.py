"""
======================================================================================
 Simulation of the Multi-Type Moran Process - Undergraduate Final Year Project 2024/25
======================================================================================
 Author      : Thomas Louis Sigone (BSc Computer Science, QMUL - 220640370)
 Supervisor  : Dr Marc Roth
 Institution : School of Electronic Engineering & Computer Science,
               Queen Mary University of London
 Date        : 06 May 2025
 File        : Moran_Process_Simulation_V3.5.py

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------
This application is the culmination of my undergraduate final-year project.  
It provides an interactive, extensible environment for exploring the multi-type
Moran process on arbitrary graphs, going well beyond the classic two-type
setting.
Key capabilities include:

- Graph generation / import
    - random (GNM, Erdos-Rényi), spring-layout positioning, CSV import/export  
- Mutation management
    - unlimited mutation “types”, per-type fitness, colour & shape, colour-blind
      palette, run-time assignment, scripted batch editing  
Dynamic simulation
    - single-step, x10, simulation fully automated with non-linear speed control,
      draggable nodes, edge creation/deletion, node-level context menu  
Analytics & visualisation
    - real-time state counts, average fitness, scrollable info panel, per-node
    inspector, Monte-Carlo fixation statistics with progress bar & pie chart  
User experience
    - PyQt5 GUI, high-DPI aware, keyboard-free operation, colour-blind mode,
      accessible shapes
Data I/O
    - one-click CSV import/export (nodes + edges + styling), graph reset, graph
      size editor, mutation config script editor

The software is intended for researchers, educators and students who need to
simulate or demonstrate the Multi-Type Moran Process.

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------
1. Create & activate the Conda environment
The `environment.yml` shipped with the repo is cross-platform; Conda automatically selects the correct build for your OS.

Unix / macOS (bash/zsh)

$ conda env create -f environment.yml
$ conda activate ECS635U

Windows (PowerShell / CMD)

:: Open "Anaconda Prompt" *or* run `conda init powershell` once.
PS> mamba env create -f environment.yml
PS> conda activate ECS635U

2. Run the application:

$ python -u ./src/Moran_Process_Simulation_V3.5.py

3. Use the toolbar buttons to edit the graph and start the simulation.
    Right-click on a node for context actions; double-click blank space
    to add a new node; left click to select a node and view its properties.

--------------------------------------------------------------------------------
SOFTWARE REQUIREMENTS
--------------------------------------------------------------------------------
• Python 3.10 +           • NumPy ≥ 1.26         • NetworkX ≥ 3.2  
• Matplotlib ≥ 3.8       • PyQt5 ≥ 5.15         • Pandas ≥ 2.1  

The tool has been tested on Windows 11, macOS 14 and Ubuntu 22.04 LTS.  High-DPI
monitors are fully supported via Qt’s automatic scaling.

--------------------------------------------------------------------------------
ACKNOWLEDGEMENTS
--------------------------------------------------------------------------------
This project builds on algorithmic insights from Goldberg, Roth, and Schwarz 
(2024), "Parameterized Approximation of the Fixation Probability of the Dominant 
Mutation in the Multi‑Type Moran Process", and the foundational model introduced 
by Moran (1958), "Random processes in genetics."

I would like to express my deepest gratitude to Dr Marc Roth for his outstanding 
guidance, expertise, and encouragement throughout this project. His support and 
insights have been instrumental in shaping my work and in motivating me to reach 
my highest academic potential. I would also like to thank the teaching staff at 
the School of Electronic Engineering and Computer Science at QMUL for their 
continuous support and dedication.

I am profoundly grateful to my grandfather; without his absolute support and 
trust, rigorous education, and exemplary character, I would not be where I am 
today. Fulfilling my dream of studying in the UK was only part of the journey 
- I later understood that my true dream was to make him proud. Thank you for 
always believing in me and supporting my aspirations. I miss you deeply, Nonno. 
To my grandmother, for her boundless sweetness and kindness. 
To my parents, for raising me with love, strength, and unwavering belief in my
potential. 
To my sister, my lighthouse in London, whose presence has brought me clarity and 
comfort. To the rest of my family, for all the love and the foundation you've 
given me. 
To my friends for their encouragement, laughter, and support during challenging 
moments.
Your presence has made this journey lighter and more meaningful.

--------------------------------------------------------------------------------
LICENCE
--------------------------------------------------------------------------------
© 2025 Thomas Louis Sigone.  Released under the MIT Licence.  See LICENCE file
for full text.  Citation of this software in academic work is appreciated:

  Sigone, T. L. (2025) *Multi-Type Moran Process Simulation Tool* (Version 3.4)
  [Computer software].  Queen Mary University of London.

================================================================================
"""


# Initial configuration: mutation_1=10,2.0; mutation_2=5,5.0; mutation_3=10,3.0

import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_MAC_WANTS_LAYER"] = "1"

import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg so that all windows use Qt

import numpy as np
import networkx as nx
import random
import colorsys
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.collections import PathCollection
import copy  # For deep copying the graph
import re
import pandas as pd
from pathlib import Path


# --- Import PyQt5 for the editors and timers ---
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QCursor
import sys

# Enable high-DPI scaling for Qt
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# --- Global Graph Size and Layout Parameters ---
default_total_nodes = 50
default_total_edges = 100
default_spacing = 0.9  # Set default spacing high

# Global flag to indicate whether we are in colour-blind mode.
COLOR_BLINDED_MODE = False

# --- State Colors Configuration ---
ORIGINAL_STATE_COLORS = {
    "healthy": "blue",
    "mutation_1": "red",
    "mutation_2": "green",
    # Additional mutations will be assigned random pastel colours.
}

# --- State Shapes Configuration ---
ORIGINAL_STATE_SHAPES = {
    "healthy": "o",       # circle
    "mutation_1": "s",    # square
    "mutation_2": "h",    # hexagon
    # Additional states will be assigned if not already defined.
}

# Info box global variables
MAX_INFO_LINES   = 6          # how many lines can be shown at once
info_scroll_idx  = 0           # top‑most line currently displayed
last_info_lines  = []          # cache of all lines from the last draw

#--- Original State Colors and Shapes ---
STATE_COLORS = ORIGINAL_STATE_COLORS.copy()
STATE_SHAPES = ORIGINAL_STATE_SHAPES.copy()
PREVIOUS_STATE_COLORS = {}
SAFE_SHAPES = ['s', '^', 'D', 'p', 'h', 'H', '8', 'd','P', 'X', '<', '>', 'v', 'o'] 

def get_shape_for_state(state: str) -> str:
    """
    Return (and memo-ise) a unique shape for this state.
    Called from visualize_graph(), so new mutations that appear *after*
    the box was ticked still get a symbol.
    """
    if state in STATE_SHAPES:
        return STATE_SHAPES[state]

    # find the first free symbol that isn't yet used
    taken = set(STATE_SHAPES.values())
    for sym in SAFE_SHAPES:
        if sym not in taken:
            STATE_SHAPES[state] = sym
            break
    else:                           # ran out → fall back to circle
        STATE_SHAPES[state] = 'o'

    return STATE_SHAPES[state]

def recolor_button(btn: QtWidgets.QPushButton, hex_colour: str):
    btn.setStyleSheet(f"background-color: {hex_colour};")
    btn.setProperty("hex", hex_colour)          # ← save it here
    btn.style().unpolish(btn)
    btn.style().polish(btn)
    btn.update()

def get_random_color():
    """Return a random vivid colour with high saturation/brightness."""
    h = random.random()          # 0‑1
    s = 0.9                      # keep it colourful
    v = 0.9
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def get_color_for_state(state):
    """
    Return the color for a given state.
    If the state is not defined in STATE_COLORS, assign a random pastel color and store it.
    """
    if state in STATE_COLORS:
        return STATE_COLORS[state]
    else:
        color = get_random_color()
        STATE_COLORS[state] = color
        return color

# --- Node-appearance toggles ---
SHOW_NODE_LABELS = True          # controls nx.draw_networkx_labels
USE_SHAPES       = False          # controls whether per-state shapes are used

# --- Default Mutation Configuration ---
default_config_str = "mutation_1=10,2.0; mutation_2=5,5.0; mutation_3=10,3.0"

def parse_mutation_config(text):
    """
    Parse a mutation configuration string using the notation:
      mutation_1=10,2.0; mutation_2=5,2.5
    Each entry is separated by a semicolon; within each entry, the mutation name is
    separated by '=' from the count and fitness (which are separated by a comma).
    """
    mutations = []
    entries = text.split(';')
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        try:
            name_part, params = entry.split('=')
            name = name_part.strip()
            count_str, fitness_str = params.split(',')
            mutations.append({
                "name": name,
                "count": int(count_str.strip()),
                "fitness": float(fitness_str.strip())
            })
        except Exception as e:
            print(f"Error parsing '{entry}':", e)
    return mutations

mutation_configs = parse_mutation_config(default_config_str)

# Store the initial graph here (after reinitialization)
initial_graph = None
initial_pos = None  # Store the initial node positions

# Global variable for the currently selected node.
selected_node = None

# --- Graph Initialization ---
def initialize_graph(n, k):
    graph = nx.gnm_random_graph(n, k)
    for node in graph.nodes():
        graph.nodes[node]['fitness'] = 1  # Baseline fitness.
        graph.nodes[node]['state'] = "healthy"
    return graph

# --- Mutation Initialization ---
def initialize_mutations(graph, mutation_configs, healthy_fitness=1):
    available_nodes = list(graph.nodes())
    for node in available_nodes:
        graph.nodes[node]['state'] = "healthy"
        graph.nodes[node]['fitness'] = healthy_fitness
    for mutation in mutation_configs:
        m_name = mutation.get("name", "mutation")
        m_count = mutation.get("count", 1)
        m_fitness = mutation.get("fitness", 1.5)
        if m_count > len(available_nodes):
            m_count = len(available_nodes)
        selected_nodes = random.sample(available_nodes, m_count)
        for node in selected_nodes:
            graph.nodes[node]['state'] = m_name
            graph.nodes[node]['fitness'] = m_fitness
        available_nodes = [node for node in available_nodes if node not in selected_nodes]

# --- Moran Process Step ---
def moran_step(graph):
    """
    Perform one Moran process step.
    Returns a tuple: (graph, changed)
    where changed is True if the target node changed state.
    """
    vertices = list(graph.nodes())
    weights = [graph.nodes[node]['fitness'] for node in vertices]
    v = random.choices(vertices, weights=weights, k=1)[0]
    neighbors = list(graph.neighbors(v))
    if not neighbors:
        print(f"No action for node {v} as it has no neighbors.")
        return graph, False
    u = random.choice(neighbors)
    print(f"Node {v} (state: {graph.nodes[v]['state']}) selected for reproduction.")
    print(f"Node {u} (state: {graph.nodes[u]['state']}) is affected.")
    old_state = graph.nodes[u]['state']
    graph.nodes[u]['state'] = graph.nodes[v]['state']
    graph.nodes[u]['fitness'] = graph.nodes[v]['fitness']
    changed = (old_state != graph.nodes[u]['state'])
    if changed:
        print(f"Node {u} changed state from {old_state} to {graph.nodes[u]['state']} due to node {v}.")
    else:
        print(f"Node {u} remains in state {graph.nodes[u]['state']} after influence from node {v}.")
    return graph, changed

# Helper functions for clearing axes
def wipe_axes(ax):
    """Clear *ax* but keep the frame visible."""
    ax.clear()
    ax.set_xticks([])            # no ticks
    ax.set_yticks([])
    ax.set_frame_on(True)        # make sure the spines are drawn

def wipe_info_panel(ax):
    """Clear ax but leave it frameless (used after Clear)."""
    ax.clear()
    ax.axis('off')        # no spines, no ticks

# helper – programmatically change the knob without firing the callback (prevents infinite recursion)
def safe_set_slider(val):
    was = scroll_slider.eventson
    scroll_slider.eventson = False       # temporarily mute callbacks
    scroll_slider.set_val(val)
    scroll_slider.eventson = was

# --- Visualization ---
def visualize_graph(graph, ax, info_ax, pos):
    """Redraw network and side panels, respecting SHOW_NODE_LABELS / USE_SHAPES."""
    global last_info_lines, info_scroll_idx
    ax.clear()
    
    # If the graph is empty (e.g. after Clear button)
    if graph.number_of_nodes() == 0:
        wipe_axes(ax)             # graph frame (or wipe_info_panel, …
        wipe_info_panel(info_ax)

        last_info_lines[:] = ["Graph is empty - double-click to add nodes."]
        info_scroll_idx    = 0

        # HIDE the scrollbar completely and put the knob back at the top
        scroll_bar_ax.set_visible(False) 
        scroll_slider.ax.set_visible(False)
        safe_set_slider(0)                     # knob back to 0
        scroll_slider.ax.set_ylim(0, 1)        # keeps limits distinct

        plt.draw()
        return

    # --- 1. group nodes by state
    state_groups = {}
    for n in graph.nodes():
        st = graph.nodes[n].get("state", "healthy")
        state_groups.setdefault(st, []).append(n)

    # --- 2. draw every group
    for st, nodes in state_groups.items():
        colour = get_color_for_state(st)
        shape  = get_shape_for_state(st) if USE_SHAPES else "o"

        nx.draw_networkx_nodes(
            graph, pos=pos, ax=ax,
            nodelist=nodes, node_color=colour, node_shape=shape
        )

    nx.draw_networkx_edges(graph, pos=pos, ax=ax)

    if SHOW_NODE_LABELS:
        # prettify:  3.0 → "3",  otherwise str(node)
        labels = {
            n: str(int(n)) if isinstance(n, float) and n.is_integer() else str(n)
            for n in graph.nodes()
        }
        nx.draw_networkx_labels(graph, pos=pos, labels=labels, ax=ax)

    # --- 3. build text lines for the info panel

    state_summary = {}
    for n in graph.nodes():
        st  = graph.nodes[n]['state']
        fit = graph.nodes[n]['fitness']
        entry = state_summary.setdefault(st, {"count": 0, "fitness": None})
        entry["count"] += 1
        if st != "healthy" and entry["fitness"] is None:
            entry["fitness"] = fit

    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    avg_fit     = sum(graph.nodes[n]['fitness'] for n in graph.nodes()) / total_nodes

    lines = [
        f"Nodes: {total_nodes},  Edges: {total_edges}",
        f"Average Overall Fitness: {avg_fit:.2f}",
    ]
    for st, info in state_summary.items():
        if st == "healthy":
            lines.append(f"{st}: {info['count']}")
        else:
            lines.append(f"{st}: {info['count']}  (fitness {info['fitness']:.2f})")

    # -- 3a. store them for the scroll handler
    last_info_lines[:] = lines

    # --- 4. keep the slider in‑sync
    new_max = max(1, len(lines) - MAX_INFO_LINES)
    if new_max != scroll_slider.valmax:
        scroll_slider.valmax = new_max
        scroll_slider.ax.set_ylim(scroll_slider.valmin, new_max)
        if info_scroll_idx > new_max:          # clamp only if necessary
            info_scroll_idx = new_max
            scroll_slider.set_val(info_scroll_idx)
    
    # --- 5. display the visible slice
    visible = lines[info_scroll_idx : info_scroll_idx + MAX_INFO_LINES]

    info_ax.clear(); info_ax.axis("off")
    info_ax.text(
        0.02, 0.5, "\n".join(visible),
        transform=info_ax.transAxes, ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor="black")
    )

    if graph.number_of_nodes() <= 1:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    else:
        ax.autoscale()          # keeps normal behaviour for bigger graphs

    # --- 6. scroll bar: show / hide and update range
    n_lines   = len(lines)
    need_bar  = n_lines > MAX_INFO_LINES

    # show / hide the bar itself
    scroll_bar_ax.set_visible(need_bar)
    scroll_slider.ax.set_visible(need_bar)

    # update range only when we need the bar
    if need_bar:
        new_max = n_lines - MAX_INFO_LINES          # ≥ 1 here
        if scroll_slider.valmax != new_max:
            scroll_slider.valmax = new_max
            scroll_slider.ax.set_ylim(new_max, 0)   # upper, lower
    else:
        # hide the knob at the top and reset range to 1 so limits differ
        safe_set_slider(0)
        scroll_slider.valmax = 1
        scroll_slider.ax.set_ylim(0, 1)

    plt.draw()

# --- Node Info Panel Update ---
orig_node_info_ax_pos = [0.2115, 0.016, 0.15, 0.08] # Original position
def update_node_info(node):
    info_text = (f"Node {node}\n"
                 f"State: {current_graph.nodes[node]['state']}\n"
                 f"Fitness: {current_graph.nodes[node]['fitness']:.2f}\n"
                 f"Neighbours: {len(list(current_graph.neighbors(node)))}")
    node_info_ax.clear()
    node_info_ax.axis('off')
    node_info_ax.set_position(orig_node_info_ax_pos)
    node_info_ax.text(0.02, 0.5, info_text, transform=node_info_ax.transAxes,
                      ha='left', va='center',
                      bbox=dict(facecolor='white', edgecolor='black'))
    plt.draw()

# --- Pick Event Handler ---
def on_pick(event):
    global selected_node
    if isinstance(event.artist, PathCollection):
        ind = event.ind[0]
        selected_node = list(current_graph.nodes())[ind]
        update_node_info(selected_node)
        print(f"Selected node: {selected_node}")

# --- Click on White Space Handler ---
def on_click(event):
    global selected_node, edge_source_node
    if event.inaxes != ax:
        return
    clicked = np.array([event.xdata, event.ydata])
    if event.button == 3:
        for node, coord in pos.items():
            if np.linalg.norm(clicked - np.array(coord)) < click_threshold:
                if edge_source_node is not None and edge_source_node != node:
                    if not current_graph.has_edge(edge_source_node, node):
                        current_graph.add_edge(edge_source_node, node)
                        print(f"Added edge from {edge_source_node} to {node}")
                    else:
                        print("Edge already exists.")
                    edge_source_node = None
                    visualize_graph(current_graph, ax, info_ax, pos)
                    return
                else:
                    show_node_context_menu(node, event)
                    return
    if event.button == 1:
        for node, coord in pos.items():
            if np.linalg.norm(clicked - np.array(coord)) < click_threshold:
                selected_node = node
                update_node_info(selected_node)
                print(f"Selected node: {node}")
                return
        selected_node = None
        node_info_ax.clear()
        node_info_ax.axis('off')
        node_info_ax.set_position(orig_node_info_ax_pos)
        plt.draw()
 
# --- Context Menu for Node ---
def show_node_context_menu(node, event):
    from PyQt5.QtWidgets import QMenu, QInputDialog
    menu = QMenu()
    add_edge_action = menu.addAction("Add edge from here")
    assign_mutation_action = menu.addAction("Assign Mutation")
    delete_node_action = menu.addAction("Delete Node")
    action = menu.exec_(QCursor.pos())
    
    if action == add_edge_action:
        global edge_source_node
        edge_source_node = node
        print(f"Selected node {node} as the source for a new edge. Now right-click a different node to connect.")
    
    elif action == assign_mutation_action:
        mutations = list(STATE_COLORS.keys())
        mutations.append("New Mutation...")
        mutation, ok = QInputDialog.getItem(None, "Assign Mutation", "Select mutation:", mutations, 0, False)
        if ok and mutation:
            if mutation == "New Mutation...":
                new_mutation, ok_name = QInputDialog.getText(None, "New Mutation", "Enter new mutation name:")
                if ok_name and new_mutation:
                    new_fitness, ok_fitness = QInputDialog.getDouble(None, "New Mutation", 
                                                                     "Enter fitness value:", 
                                                                     1.5, 0.1, 10.0, 2)
                    if ok_fitness:
                        current_graph.nodes[node]['state'] = new_mutation
                        current_graph.nodes[node]['fitness'] = new_fitness
                        get_color_for_state(new_mutation)
                        visualize_graph(current_graph, ax, info_ax, pos)
                        print(f"Node {node} now has mutation '{new_mutation}'.")
            else:
                current_graph.nodes[node]['state'] = mutation
                current_graph.nodes[node]['fitness'] = 1.5 if mutation != "healthy" else 1
                visualize_graph(current_graph, ax, info_ax, pos)
                print(f"Node {node} now has mutation '{mutation}'.")
    
    elif action == delete_node_action:
        current_graph.remove_node(node)
        pos.pop(node, None)
        visualize_graph(current_graph, ax, info_ax, pos)
        print(f"Deleted node: {node}")

# --- Control Functions ---
step_count = 0

def step(event):
    global current_graph, step_count
    current_graph, _ = moran_step(current_graph)
    step_count += 1
    update_step_label()
    visualize_graph(current_graph, ax, info_ax, pos)
    if selected_node is not None:
        update_node_info(selected_node)

def step_x10(event):
    global current_graph, step_count
    for _ in range(10):
        current_graph, _ = moran_step(current_graph)
    step_count += 10
    update_step_label()
    visualize_graph(current_graph, ax, info_ax, pos)
    if selected_node is not None:
        update_node_info(selected_node)

def update_step_label():
    # hide when 0, as before
    if step_count == 0:
        step_label_ax.set_visible(False)
        fig.canvas.draw_idle()
        return

    # --- where to put the label ---
    info_pos   = info_ax.get_position()         # (x0, y0, w, h)
    margin     = 0.005
    new_height = 0.04

    # place it *just below* the info panel – use y1 (top), not y0 (bottom)
    new_y0 = max(info_pos.y0 - new_height - margin,          # below if room
                 margin)                                     # otherwise clamp
    step_label_ax.set_position([0.01, new_y0, 0.18, new_height])

    # --- draw ---
    step_label_ax.set_visible(True)
    step_label_ax.clear();  step_label_ax.axis('off')
    step_label_ax.text(0.02, 0.5, f"Number of steps: {step_count}",
                       transform=step_label_ax.transAxes,
                       fontsize=10, ha='left', va='center')
    fig.canvas.draw_idle()

def change_fitness(delta):
    global current_graph
    for node in current_graph.nodes():
        if current_graph.nodes[node]['state'] != "healthy":
            new_fitness = current_graph.nodes[node]['fitness'] + delta
            current_graph.nodes[node]['fitness'] = max(0.1, new_fitness)
    visualize_graph(current_graph, ax, info_ax, pos)
    if selected_node is not None:
        update_node_info(selected_node)

def increase_fitness(event):
    change_fitness(0.1)

def decrease_fitness(event):
    change_fitness(-0.1)

# --- Reset Functions ---
def reinit_graph():
    global current_graph, pos, step_count, initial_graph, initial_pos, simulation_running, selected_node
    simulation_running = False  # Stop any running simulation.
    total_nodes = default_total_nodes
    total_edges = default_total_edges
    current_graph = initialize_graph(total_nodes, total_edges)
    initialize_mutations(current_graph, mutation_configs, healthy_fitness=1)
    pos = nx.spring_layout(current_graph, k=default_spacing)
    initial_pos = copy.deepcopy(pos)  # Store the initial positions.
    step_count = 0
    update_step_label()
    visualize_graph(current_graph, ax, info_ax, pos)
    initial_graph = copy.deepcopy(current_graph)
    selected_node = None

def reset_graph():
    global current_graph, pos, step_count, initial_graph, initial_pos, simulation_running, selected_node
    simulation_running = False
    current_graph = copy.deepcopy(initial_graph)
    pos = copy.deepcopy(initial_pos)
    step_count = 0
    update_step_label()
    visualize_graph(current_graph, ax, info_ax, pos)
    simulation_delay_slider.set_val(0.315)  # Reset slider to default (≈100ms)
    selected_node = None
    node_info_ax.clear()
    node_info_ax.axis('off')
    node_info_ax.set_position(orig_node_info_ax_pos)
    plt.draw()

# --- PyQt5 Editors for Config & Graph Size ---
def update_graph_size(new_nodes, new_edges):
    global current_graph, pos, initial_pos
    current_node_count = len(current_graph.nodes())
    if new_nodes > current_node_count:
        extra = new_nodes - current_node_count
        print(f"Adding {extra} new node(s).")
        for i in range(current_node_count, current_node_count + extra):
            current_graph.add_node(i)
            initial_pos[i] = [random.uniform(-1, 1), random.uniform(-1, 1)]
        pos.update({i: initial_pos[i] for i in range(current_node_count, current_node_count + extra)})
        for i in range(current_node_count, current_node_count + extra):
            if current_graph.degree(i) == 0:
                other = random.choice(list(current_graph.nodes()))
                while other == i:
                    other = random.choice(list(current_graph.nodes()))
                current_graph.add_edge(i, other)
    elif new_nodes < current_node_count:
        nodes_to_remove = list(range(new_nodes, current_node_count))
        print(f"Removing {len(nodes_to_remove)} node(s): {nodes_to_remove}")
        current_graph.remove_nodes_from(nodes_to_remove)
        for i in nodes_to_remove:
            pos.pop(i, None)
            initial_pos.pop(i, None)
    all_nodes = list(current_graph.nodes())
    while len(current_graph.edges()) < new_edges:
        u, v = random.sample(all_nodes, 2)
        if not current_graph.has_edge(u, v):
            current_graph.add_edge(u, v)
    while len(current_graph.edges()) > new_edges:
        edge = random.choice(list(current_graph.edges()))
        current_graph.remove_edge(*edge)
    initialize_mutations(current_graph, mutation_configs, healthy_fitness=1)
    visualize_graph(current_graph, ax, info_ax, pos)
    
def open_graph_size_editor(initial_nodes, initial_edges):
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Set Graph Size")
    layout = QtWidgets.QVBoxLayout(dialog)
    form_layout = QtWidgets.QFormLayout()
    spin_nodes = QtWidgets.QSpinBox()
    spin_nodes.setRange(1, 10000)
    spin_nodes.setValue(initial_nodes)
    spin_edges = QtWidgets.QSpinBox()
    spin_edges.setRange(0, 100000)
    spin_edges.setValue(initial_edges)
    form_layout.addRow("Number of Nodes:", spin_nodes)
    form_layout.addRow("Number of Edges:", spin_edges)
    layout.addLayout(form_layout)
    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    layout.addWidget(button_box)
    def accept():
        global default_total_nodes, default_total_edges
        new_nodes = spin_nodes.value()
        new_edges = spin_edges.value()
        default_total_nodes = new_nodes
        default_total_edges = new_edges
        if new_nodes == len(current_graph.nodes()):
            update_graph_size(new_nodes, new_edges)
        else:
            reinit_graph()
        dialog.accept()
    def reject():
        dialog.reject()
    button_box.accepted.connect(accept)
    button_box.rejected.connect(reject)
    dialog.resize(300, 150)
    dialog.exec_()

def pick_color(btn):
    from PyQt5.QtWidgets import QColorDialog
    c = QColorDialog.getColor()
    if c.isValid():
        btn.setStyleSheet("background-color: " + c.name())

# --- Configuration Editor ---
def open_config_editor_text(initial_text=default_config_str):
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Edit Mutation Config (Script)")
    layout = QtWidgets.QVBoxLayout(dialog)
    text_edit = QtWidgets.QTextEdit()
    text_edit.setPlainText(initial_text)
    layout.addWidget(text_edit)
    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    layout.addWidget(button_box)
    def accept():
        config_text = text_edit.toPlainText().strip()
        apply_config_callback(config_text)
        dialog.accept()
    def reject():
        dialog.reject()
    button_box.accepted.connect(accept)
    button_box.rejected.connect(reject)
    # Increase width from 500 to 700 pixels for better horizontal readability.
    dialog.resize(700, 300)
    dialog.exec_()

# --- Edit config button ---
def open_config_editor():
    """Open the “Edit Mutation Config” dialog."""
    import re
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QSpinBox,
        QDoubleSpinBox, QPushButton, QColorDialog, QLabel, QMessageBox,
        QCheckBox
    )

    dialog = QDialog()
    dialog.setWindowTitle("Edit Mutation Config")
    layout = QVBoxLayout(dialog)

    # 1.  Mutation rows (name, count, fitness, colour)
    rows_widget  = QWidget()
    rows_layout  = QVBoxLayout(rows_widget)
    mutation_rows = []

    def add_row(name="", count=1, fitness=1.5, color="#ff9999"):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        name_edit   = QLineEdit(name)
        count_spin  = QSpinBox();      count_spin.setRange(1, 10000);  count_spin.setValue(count)
        fitness_spin= QDoubleSpinBox();fitness_spin.setRange(0.1, 10.0);fitness_spin.setSingleStep(0.1);fitness_spin.setValue(fitness)

        color_button= QPushButton()
        color_button.setFixedWidth(40)
        recolor_button(color_button, color) 


        def pick_color():
            c = QColorDialog.getColor()
            if c.isValid():
                recolor_button(color_button, c.name())
        color_button.clicked.connect(pick_color)

        remove_button = QPushButton("Remove")
        def remove_row():
            rows_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            mutation_rows.remove(row_data)
        remove_button.clicked.connect(remove_row)

        for w in (
            QLabel("Name:"), name_edit,
            QLabel("Count:"), count_spin,
            QLabel("Fitness:"), fitness_spin,
            QLabel("Color:"), color_button,
            remove_button
        ):
            row_layout.addWidget(w)

        rows_layout.addWidget(row_widget)
        row_data = {"widget": row_widget, "name": name_edit, "count": count_spin,
                    "fitness": fitness_spin, "color": color_button}
        mutation_rows.append(row_data)

    # pre-populate from current mutation_configs
    for i, m in enumerate(mutation_configs):
        c = OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)] if COLOR_BLINDED_MODE \
            else STATE_COLORS.get(m["name"], get_random_color())
        add_row(m["name"], m["count"], m["fitness"], c)

    layout.addWidget(rows_widget)

    # button to add new mutation row
    add_mutation_button = QPushButton("Add Mutation")
    def on_add_mutation():
        idx = len(mutation_rows)
        colour = OKABE_ITO_COLORS[idx % len(OKABE_ITO_COLORS)] if COLOR_BLINDED_MODE else get_random_color()
        add_row(f"Mutation_{idx+1}", 1, 1.5, colour)
    add_mutation_button.clicked.connect(on_add_mutation)
    layout.addWidget(add_mutation_button)

    # 2.  Colour-blind / normal mode buttons
    mode_layout = QHBoxLayout()
    convert_cb_button  = QPushButton("Convert to Colour-blind Mode")
    revert_normal_button = QPushButton("Revert to Normal Mode")
    convert_cb_button.setStyleSheet("background-color:#FFCC00")
    revert_normal_button.setStyleSheet("background-color:#CCCCCC")
    mode_layout.addWidget(convert_cb_button)
    mode_layout.addWidget(revert_normal_button)
    layout.addLayout(mode_layout)

    # preview-refresh helpers
    def update_color_previews():
        for i, row in enumerate(mutation_rows):
            new_c = OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)]
            recolor_button(row["color"], new_c)
            STATE_COLORS[row["name"].text().strip()] = new_c

    def update_normal_previews():
        for row in mutation_rows:
            name = row["name"].text().strip()
            recolor_button(row["color"], STATE_COLORS.get(name, get_random_color()))

    # 3.  Appearance check-boxes
    check_layout = QHBoxLayout()
    label_cb = QCheckBox("Show node labels");  label_cb.setChecked(SHOW_NODE_LABELS)
    shape_cb = QCheckBox("Use distinct shapes");shape_cb.setChecked(USE_SHAPES)
    check_layout.addWidget(label_cb); check_layout.addWidget(shape_cb)
    layout.addLayout(check_layout)

    def on_label_toggle(state):
        global SHOW_NODE_LABELS
        SHOW_NODE_LABELS = bool(state)
        visualize_graph(current_graph, ax, info_ax, pos)

    def on_shape_toggle(state):
        global USE_SHAPES
        USE_SHAPES = bool(state)
        get_shape_for_state(state)
        visualize_graph(current_graph, ax, info_ax, pos)

    label_cb.stateChanged.connect(on_label_toggle)
    shape_cb.stateChanged.connect(on_shape_toggle)

    # colour-blind button: tick the box *and* assign shapes
    def on_convert_to_cb():
        enable_colour_blind_mode()
        update_color_previews()
        shape_cb.setChecked(True)
        for m in [m["name"] for m in mutation_configs]:
            get_shape_for_state(m)
        visualize_graph(current_graph, ax, info_ax, pos)

    convert_cb_button.clicked.connect(on_convert_to_cb)

    revert_normal_button.clicked.connect(
        lambda: (revert_normal_mode(), update_normal_previews())
    )

    # 4.  Bottom row: Input-script / OK / Cancel
    bottom_layout = QHBoxLayout()
    input_script_button = QPushButton("Input Script")
    ok_button     = QPushButton("OK")
    cancel_button = QPushButton("Cancel")
    bottom_layout.addWidget(input_script_button)
    bottom_layout.addStretch()
    bottom_layout.addWidget(cancel_button)
    bottom_layout.addWidget(ok_button)
    layout.addLayout(bottom_layout)
    input_script_button.clicked.connect(lambda: open_config_editor_text())

    # OK-handler: write back configs
    def on_ok():
        # --- 1. collect rows into a fresh config list
        new_configs = []
        for i, row in enumerate(mutation_rows):
            name = row["name"].text().strip()
            if not name:
                continue

            count   = row["count"].value()
            fitness = row["fitness"].value()

            # colour depends on the current mode
            if COLOR_BLINDED_MODE:
                colour = OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)]
            else:
                colour = row["color"].property("hex") or get_random_color()

            new_configs.append(
                {"name": name, "count": count, "fitness": fitness, "color": colour}
            )

        if not new_configs:
            QMessageBox.warning(dialog, "Invalid Input", "Please add at least one mutation.")
            return

        # --- 2. update global mutation list and colour map
        global mutation_configs
        mutation_configs = [
            {"name": c["name"], "count": c["count"], "fitness": c["fitness"]}
            for c in new_configs
        ]
        for c in new_configs:
            STATE_COLORS[c["name"]] = c["color"]

        # --- 3. let the common helper handle sizing + re‑init
        config_text = "; ".join(f"{m['name']}={m['count']},{m['fitness']}"
                                for m in mutation_configs)

        # this grows the graph if required, re‑initialises mutations,
        # updates default_config_str, and redraws everything
        apply_config_callback(config_text)

        dialog.accept()

    ok_button.clicked.connect(on_ok)
    cancel_button.clicked.connect(dialog.reject)

    dialog.resize(700, 400)
    dialog.exec_()

# --- Apply Config Callback ---
def apply_config_callback(new_config_text):
    global mutation_configs, default_config_str, current_graph, pos, initial_pos, default_total_nodes
    new_configs = parse_mutation_config(new_config_text)
    if not new_configs:
        print("Failed to parse the mutation configuration.")
        return
    if new_config_text.strip() == default_config_str.strip():
        print("Configuration unchanged. Not updating graph.")
        return
    required_nodes = sum(m["count"] for m in new_configs)
    current_nodes = len(current_graph.nodes())
    if required_nodes > current_nodes:
        extra = required_nodes - current_nodes
        print(f"New configuration requires {required_nodes} nodes (current: {current_nodes}). Adding {extra} new node(s).")
        for i in range(current_nodes, current_nodes + extra):
            current_graph.add_node(i)
            initial_pos[i] = [random.uniform(-1, 1), random.uniform(-1, 1)]
        pos.update({i: initial_pos[i] for i in range(current_nodes, current_nodes + extra)})
        for i in range(current_nodes, current_nodes + extra):
            if current_graph.degree(i) == 0:
                other = random.choice(list(current_graph.nodes()))
                while other == i:
                    other = random.choice(list(current_graph.nodes()))
                current_graph.add_edge(i, other)
    else:
        print("New configuration does not require extra nodes. Keeping current node positions.")
    mutation_configs = new_configs
    default_config_str = new_config_text
    initialize_mutations(current_graph, mutation_configs, healthy_fitness=1)
    visualize_graph(current_graph, ax, info_ax, pos)
    default_total_nodes = len(current_graph.nodes())

def on_resize(event):
    update_step_label()

# --- Simulation Mode ---
simulation_running = False

def run_simulation():
    global simulation_running
    if simulation_running:
        return  # Prevent multiple simulation runs.
    simulation_running = True
    simulation_loop()

def simulation_loop():
    global simulation_running, current_graph, step_count, selected_node
    if not simulation_running:
        print("Simulation aborted.")
        return
    distinct_states = set(nx.get_node_attributes(current_graph, 'state').values())
    if len(distinct_states) <= 1:
        simulation_running = False
        print("Simulation finished: fixation reached.")
        return
    current_graph, changed = moran_step(current_graph)
    step_count += 1
    if changed:
        visualize_graph(current_graph, ax, info_ax, pos)
        update_step_label()
        if selected_node is not None:
            update_node_info(selected_node)
        delay = slider_to_delay(simulation_delay_slider.val)
        QtCore.QTimer.singleShot(delay, simulation_loop)
    else:
        QtCore.QTimer.singleShot(0, simulation_loop)

def stop_simulation(event=None):
    global simulation_running
    simulation_running = False
    print("Simulation stopped manually.")

def clear_graph(event):
    global current_graph, pos, initial_graph, initial_pos, step_count, selected_node, simulation_running
    simulation_running = False
    current_graph = nx.Graph()
    pos = {}
    initial_graph = nx.Graph()
    initial_pos = {}
    step_count = 0
    selected_node = None
    ax.clear()
    info_ax.clear()
    node_info_ax.clear()
    step_label_ax.clear()
    node_info_ax.set_position(orig_node_info_ax_pos)
    node_info_ax.axis('off')
    visualize_graph(current_graph, ax, info_ax, pos)
    plt.draw()
    print("Graph cleared. Double-click in the canvas to add new nodes.")

# --- Set up Figure, Axes, and Buttons ---
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title('Multi-Type Moran Process Simulation V3.4')
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.21, bottom=0.12, right=0.98, top=0.97)

# -- summary panel
INFO_PANEL = [0.015, -0.04, 0.01, 0.35]  # x, y, width, height   (units = figure ‑‑ 0‑1)

info_ax = fig.add_axes(INFO_PANEL)
info_ax.axis('off')

# -- node info panel
# 1) create a tiny axes for the vertical slider
scroll_bar_ax = fig.add_axes([0.19, 0.07, 0.01, 0.13], facecolor='lightgray')

scroll_slider = Slider(
    ax=scroll_bar_ax,
    label='',
    valmin=0,
    valmax=1,       
    valinit=0,
    orientation='vertical'
)
scroll_slider.valtext.set_visible(False)
scroll_slider.poly.set_visible(False)
for attr in ('poly', 'hline', 'vline'):
    artist = getattr(scroll_slider, attr, None)
    if artist is not None:
        artist.set_visible(False)


# 2) callback to update the scroll index and re‐draw
def on_scroll_slider(val):
    """
    Slider value == the first line shown in the info panel
    (0 means very top).  No extra flipping necessary because
    the y‑axis is already inverted.
    """
    global info_scroll_idx
    max_start = max(0, len(last_info_lines) - MAX_INFO_LINES)
    info_scroll_idx = max_start - int(val)
    visualize_graph(current_graph, ax, info_ax, pos)

scroll_slider.on_changed(on_scroll_slider)

node_info_ax = fig.add_axes(orig_node_info_ax_pos) # Node info panel
node_info_ax.axis('off')
step_label_ax = fig.add_axes([0.01, 0.05, 0.10, 0.09]) # Step label (bottom left)
step_label_ax.axis('off')

button_mc_ax = fig.add_axes([0.33, 0.01, 0.11, 0.09]) # Monte Carlo
button_ax3 = fig.add_axes([0.60, 0.01, 0.11, 0.09]) # Decrease Fitness
button_ax2 = fig.add_axes([0.72, 0.01, 0.11, 0.09]) # Increase Fitness
button_ax1 = fig.add_axes([0.85, 0.01, 0.06, 0.09]) # Step
button_ax4 = fig.add_axes([0.92, 0.01, 0.06, 0.09]) # Step x10

button_size = fig.add_axes([0.01, 0.89, 0.18, 0.08]) # Set Graph Size
button_edit = fig.add_axes([0.01, 0.79, 0.18, 0.08]) # Edit Config
button_reset = fig.add_axes([0.01, 0.59, 0.18, 0.08]) # Reset
button_simulate = fig.add_axes([0.01, 0.49, 0.18, 0.08]) # Simulate
button_stop_simulate = fig.add_axes([0.01, 0.35, 0.18, 0.08]) # Stop Simulation

button_clear = Button(fig.add_axes([0.01, 0.25, 0.18, 0.08]), 'Clear') # Clear Graph
button_clear.on_clicked(clear_graph)

# --- Slider Setup (Non-linear Mapping) ---
slider_ax = fig.add_axes([0.01, 0.44, 0.18, 0.01])
simulation_delay_slider = Slider(slider_ax, '', 0, 1, valinit=0.315, valstep=0.001)
simulation_delay_slider.label.set_visible(False)
simulation_delay_slider.valtext.set_visible(False)

slider_label_ax = fig.add_axes([0.01, 0.455, 0.18, 0.02])
slider_label_ax.axis('off')
slider_label = slider_label_ax.text(0.5, 0.5, "Simulation speed: 100ms",
                                    ha="center", va="center", fontsize=10)

def slider_to_delay(x, threshold=0.315, low_max=100, high_max=1000):
    if x <= threshold:
        return int(1 + (low_max - 1) * (x / threshold)**2)
    else:
        return int(low_max + (high_max - low_max) * ((x - threshold) / (1 - threshold)))

def update_slider_label(val):
    delay = slider_to_delay(val)
    slider_label.set_text(f"Simulation speed: {delay}ms")
    fig.canvas.draw_idle()

simulation_delay_slider.on_changed(update_slider_label)

# --- GUI Network customisation ---
dragging_node = None
edge_source_node = None
click_threshold = 0.05

def on_press(event):
    global dragging_node
    if event.inaxes == ax and event.button == 1:
        clicked = np.array([event.xdata, event.ydata])
        for node, coord in pos.items():
            if np.linalg.norm(clicked - np.array(coord)) < click_threshold:
                dragging_node = node
                break

def on_motion(event):
    global dragging_node
    if dragging_node is not None and event.inaxes == ax:
        pos[dragging_node] = [event.xdata, event.ydata]
        visualize_graph(current_graph, ax, info_ax, pos)
        if selected_node == dragging_node:
            update_node_info(dragging_node)

def on_release(event):
    global dragging_node
    dragging_node = None

def on_double_click(event):
    if event.inaxes == ax and event.dblclick:
        clicked = np.array([event.xdata, event.ydata])
        for node, coord in pos.items():
            if np.linalg.norm(clicked - np.array(coord)) < click_threshold:
                return
        new_node = max(current_graph.nodes()) + 1 if current_graph.nodes() else 0
        current_graph.add_node(new_node, state="healthy", fitness=1)
        pos[new_node] = [event.xdata, event.ydata]
        visualize_graph(current_graph, ax, info_ax, pos)
        print(f"Added new node: {new_node}")

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_press_event', on_double_click)

# --- Button Setup ---
button_step = Button(button_ax1, 'Step')
button_inc = Button(button_ax2, '+ Fitness')
button_dec = Button(button_ax3, '- Fitness')
button_step_x10 = Button(button_ax4, 'Step x10')
button_set_size = Button(button_size, 'Set graph size')
button_edit_config = Button(button_edit, 'Edit configuration')
button_reset = Button(button_reset, 'Reset')
button_simulate = Button(button_simulate, 'Simulate')
button_stop_simulate = Button(button_stop_simulate, 'Stop simulation')

button_step.on_clicked(step)
button_inc.on_clicked(increase_fitness)
button_dec.on_clicked(decrease_fitness)
button_step_x10.on_clicked(step_x10)
button_set_size.on_clicked(lambda event: open_graph_size_editor(default_total_nodes, default_total_edges))
button_edit_config.on_clicked(lambda event: open_config_editor())
button_reset.on_clicked(lambda event: reset_graph())
button_simulate.on_clicked(lambda event: run_simulation())
button_stop_simulate.on_clicked(stop_simulation)

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('resize_event', on_resize)

def on_info_scroll(event):
    global info_scroll_idx
    if event.inaxes is not info_ax:
        return

    direction = 1 if event.button == 'down' else -1
    max_start = max(0, len(last_info_lines) - MAX_INFO_LINES)
    info_scroll_idx = min(max_start, max(0, info_scroll_idx + direction))

    visualize_graph(current_graph, ax, info_ax, pos)
    fig.canvas.draw_idle()          # <<–– force refresh

fig.canvas.mpl_connect('scroll_event', on_info_scroll)

# --- Okabe–Ito Palette for Colour-Blind Mode ---
OKABE_ITO_COLORS = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7"   # reddish purple
]

def enable_colour_blind_mode():
    global STATE_COLORS, PREVIOUS_STATE_COLORS, COLOR_BLINDED_MODE
    if COLOR_BLINDED_MODE:
        return                         # already active
    PREVIOUS_STATE_COLORS = STATE_COLORS.copy()
    COLOR_BLINDED_MODE = True

    # overwrite colours with Okabe–Ito
    for i, mutation in enumerate([m["name"] for m in mutation_configs]):
        STATE_COLORS[mutation] = OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)]
    visualize_graph(current_graph, ax, info_ax, pos)

def revert_normal_mode():
    global STATE_COLORS, PREVIOUS_STATE_COLORS, COLOR_BLINDED_MODE
    if not COLOR_BLINDED_MODE:
        return                         # already normal
    STATE_COLORS = PREVIOUS_STATE_COLORS.copy()
    COLOR_BLINDED_MODE = False
    visualize_graph(current_graph, ax, info_ax, pos)

# --- Monte Carlo Simulation ---
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QInputDialog, QMessageBox, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import itertools

# progress‑bar axis (initially hidden) 
pbar_ax = fig.add_axes([0.2103, 0.9395, 0.77, 0.06])  # x, y, w, h in figure coords
pbar_ax.set_xlim(0, 1)
pbar_ax.set_ylim(0, 1)
pbar_ax.axis('off')
pbar_rect = pbar_ax.barh(0, 0, height=1, color="#1f77b4")[0]  # keep handle
pbar_ax.set_visible(False)

def get_simulation_parameters():
    dialog = QDialog()
    dialog.setWindowTitle("Simulation Parameters")
    layout = QVBoxLayout(dialog)
    
    iter_layout = QHBoxLayout()
    iter_label = QLabel("Number of decided iterations:")
    iter_spin = QSpinBox()
    iter_spin.setRange(1, 1000)
    iter_spin.setValue(10)
    iter_layout.addWidget(iter_label)
    iter_layout.addWidget(iter_spin)
    layout.addLayout(iter_layout)
    
    steps_layout = QHBoxLayout()
    steps_label = QLabel("Maximum steps per iteration:")
    steps_spin = QSpinBox()
    steps_spin.setRange(1000, 1000000)
    steps_spin.setValue(10000)
    steps_layout.addWidget(steps_label)
    steps_layout.addWidget(steps_spin)
    layout.addLayout(steps_layout)
    
    btn_layout = QHBoxLayout()
    cancel_btn = QPushButton("Cancel")
    ok_btn = QPushButton("OK")
    btn_layout.addWidget(cancel_btn)
    btn_layout.addWidget(ok_btn)
    layout.addLayout(btn_layout)
    
    ok_btn.clicked.connect(dialog.accept)
    cancel_btn.clicked.connect(dialog.reject)
    
    if dialog.exec_() == QDialog.Accepted:
        return iter_spin.value(), steps_spin.value()
    else:
        return None, None

def show_diagram_dialog(result_text, outcomes, n_iterations):
    diagram_dialog = QDialog()
    diagram_dialog.setWindowTitle(f"Fixation outcomes (i = {n_iterations})")
    d_layout = QVBoxLayout(diagram_dialog)
    
    fig_diagram = plt.Figure(figsize=(5, 4))
    canvas = FigureCanvas(fig_diagram)
    ax_diagram = fig_diagram.add_subplot(111)
    
    labels = [f"{state} ({count})" for state, count in outcomes.items()]
    sizes = [count for count in outcomes.values()]
    colors = [get_color_for_state(state) for state in outcomes.keys()]
    
    ax_diagram.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax_diagram.set_title(f"Fixation outcomes (i = {n_iterations})")
    ax_diagram.axis('equal')
    
    d_layout.addWidget(canvas)
    
    btn_layout = QHBoxLayout()
    back_btn = QPushButton("Back to Results")
    close_btn = QPushButton("Close")
    btn_layout.addWidget(back_btn)
    btn_layout.addWidget(close_btn)
    d_layout.addLayout(btn_layout)
    
    def on_back():
        diagram_dialog.accept()
        show_simulation_results(result_text, outcomes, n_iterations)
    
    def on_close():
        diagram_dialog.accept()
    
    back_btn.clicked.connect(on_back)
    close_btn.clicked.connect(on_close)
    
    diagram_dialog.exec_()

def show_simulation_results(result_text, outcomes, n_iterations):
    results_dialog = QDialog()
    results_dialog.setWindowTitle("Monte Carlo Simulation Results")
    layout = QVBoxLayout(results_dialog)
    
    result_label = QLabel(result_text)
    layout.addWidget(result_label)
    
    btn_layout = QHBoxLayout()
    show_btn = QPushButton("Show Diagram")
    close_btn = QPushButton("Close")
    btn_layout.addWidget(show_btn)
    btn_layout.addWidget(close_btn)
    layout.addLayout(btn_layout)
    
    def on_show():
        results_dialog.accept()
        show_diagram_dialog(result_text, outcomes, n_iterations)
    
    def on_close():
        results_dialog.accept()
    
    show_btn.clicked.connect(on_show)
    close_btn.clicked.connect(on_close)
    
    results_dialog.exec_()

def run_monte_carlo_simulation(n_iterations=10, max_steps=200_000):
    global current_graph
    outcomes = {}
    attempts = 0

    # show bar, reset width
    pbar_ax.set_visible(True)
    pbar_rect.set_width(0)
    fig.canvas.draw_idle(); QApplication.processEvents()

    for decided in range(n_iterations):
        attempts += 1
        reinit_graph()                    
        # sanity‑check first 3 nodes
        sample = list(itertools.islice(current_graph.nodes(data=True), 3))
        print(f"[init #{attempts}] sample:", sample)

        start_id = id(current_graph)         # ensure a fresh object

        # --- fixation loop
        for steps in range(1, max_steps + 1):
            current_graph, _ = moran_step(current_graph)
            states = nx.get_node_attributes(current_graph, 'state').values()
            if len(set(states)) == 1:        # FIXATED
                final_state = next(iter(states))
                outcomes[final_state] = outcomes.get(final_state, 0) + 1
                break
        else:
            print(f"attempt {attempts}: no fixation after {max_steps:,} steps")
            decided -= 1          # repeat this slot
            continue

        # --- progress bar update
        frac = (decided + 1) / n_iterations
        pbar_rect.set_width(frac)
        fig.canvas.draw_idle(); QApplication.processEvents()

        # extra safety: graph object must have changed
        assert id(current_graph) == start_id, "graph replaced inside loop!"

    # hide bar
    pbar_ax.set_visible(False)
    fig.canvas.draw_idle(); QApplication.processEvents()

    return outcomes, decided + 1, attempts

def run_simulation_iterations(event=None):
    n_iterations, max_steps = get_simulation_parameters()
    if n_iterations is None or max_steps is None:
        return
    outcomes, decided_iterations, attempts = run_monte_carlo_simulation(n_iterations, max_steps)
    if decided_iterations == 0:
        QMessageBox.information(None, "Monte Carlo Simulation Results",
                                "No decided outcomes reached. Please try again with a higher max steps value.")
        return
    result_lines = []
    total = sum(outcomes.values())
    for state, count in outcomes.items():
        percentage = count / total * 100
        result_lines.append(f"{state} wins = {percentage:.1f}%")
    result_text = "\n".join(result_lines)
    show_simulation_results(result_text, outcomes, decided_iterations)

button_mc = Button(button_mc_ax, 'Run Monte Carlo')
button_mc.on_clicked(run_simulation_iterations)

# --- Import Graph from CSV File ---
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog, QProgressDialog
from PyQt5.QtCore    import QThread, pyqtSignal
from pathlib         import Path
import numpy as np
import copy, networkx as nx   # already imported earlier

# --- 1. worker thread
class CSVImportWorker(QThread):
    """
    Background task: read the CSV, rename columns if the user mapped them,
    build a NetworkX graph, then emit `finished` or `error`.
    """
    finished = pyqtSignal(nx.Graph, str)
    error    = pyqtSignal(str)

    def __init__(self, fname: str, mapping: dict | None = None, parent=None):
        super().__init__(parent)
        self.fname   = fname
        self.mapping = mapping or {}

    def run(self):
        try:
            df = pd.read_csv(self.fname) 
            if self.mapping:
                df = df.rename(columns=self.mapping)

            # coerce ids / endpoints to int when possible
            def maybe_int(x):
                try:
                    xi = int(float(x))
                    return xi if str(xi) == str(x).strip() else x
                except Exception:
                    return x

            if 'color' in df.columns:
                df['color'] = df['color'].replace({'': np.nan})

            for col in ('source', 'target', 'id'):
                if col in df.columns:
                    df[col] = df[col].apply(maybe_int)

            g = build_graph_from_dataframe(df)
            self.finished.emit(g, self.fname)
        except Exception as e:
            self.error.emit(str(e))

# keep references so the threads stay alive
_import_workers: list['CSVImportWorker'] = []

def build_graph_from_dataframe(df: pd.DataFrame) -> nx.Graph:
    # just build the graph assuming df already has the right column names
    g = nx.Graph()

    # --- A. mixed file 
    if {'source','target','id','state','fitness','color'}.issubset(df.columns):
        edges_part = df.dropna(subset=['source','target'])
        for _, r in edges_part[['source','target']].iterrows():
            g.add_edge(r['source'], r['target'])

        nodes_part = df.dropna(subset=['id','state','fitness'])
        for _, r in nodes_part[['id','state','fitness','color']].drop_duplicates().iterrows():
            st_colour = r['color']
            if pd.notna(st_colour):
                STATE_COLORS[r['state']] = st_colour          # remember palette
            # ── ensure the vertex exists, even if it never appears in the edge list
            if r['id'] not in g:
                g.add_node(r['id'])

            g.nodes[r['id']].update(
                state   = r['state'],
                fitness = float(r['fitness'])
            )

    # --- B. pure edge list
    elif {'source', 'target'}.issubset(df.columns):
        for _, row in df[['source', 'target']].iterrows():
            g.add_edge(row['source'], row['target'])

    # --- C. pure node table
    elif {'id', 'state', 'fitness'}.issubset(df.columns):
        for _, row in df.iterrows():
            g.add_node(row['id'],
                    state=row['state'],
                    fitness=float(row['fitness']))
    else:
        raise ValueError(
            "CSV must contain either:\n"
            "  • source,target  (edge list)\n"
            "  • id,state,fitness  (node table)\n"
            "  • or all five columns together."
        )

    return g

# 2. GUI callback 
from PyQt5.QtWidgets import QProgressDialog

def refresh_mutation_configs_from_graph(g: nx.Graph):
    """
    Recreate `mutation_configs` + `default_config_str` so the Edit‑Config
    window shows the states, counts, fitness and colours of the *current* graph.
    """
    global mutation_configs, default_config_str

    # --- collect one entry per state
    summary: dict[str, dict] = {}
    for n in g.nodes():
        st  = g.nodes[n].get("state", "healthy")
        fit = g.nodes[n].get("fitness", 1.0)
        entry = summary.setdefault(st, {"count": 0, "fitness": fit})
        entry["count"] += 1
        # keep the first fitness value we encounter for that state

    # --- rebuild the global list
    mutation_configs = [
        {"name": st,
         "count": info["count"],
         "fitness": info["fitness"]}
        for st, info in summary.items()
        if st != "healthy"                      # dialog only lists mutations
    ]

    # --- rebuild the default_config_str
    default_config_str = "; ".join(
        f"{m['name']}={m['count']},{m['fitness']}"
        for m in mutation_configs
    )

def import_graph(event=None):
    global _import_workers

    fname, _ = QFileDialog.getOpenFileName(
        None, "Import graph (CSV)", "", "CSV files (*.csv)"
    )
    if not fname:
        return

    df0  = pd.read_csv(fname, nrows=0)        # header only
    cols = df0.columns.tolist()

    # --- mapping dialog
    mapping = {}
    if 'source' not in cols:
        col, ok = QInputDialog.getItem(None, "Map column",
                                       "Select column for EDGE SOURCE:", cols, 0, False)
        if ok: mapping[col] = 'source'
    if 'target' not in cols:
        col, ok = QInputDialog.getItem(None, "Map column",
                                       "Select column for EDGE TARGET:", cols, 0, False)
        if ok: mapping[col] = 'target'
    if any(k not in cols for k in ('id', 'state', 'fitness')):
        for want in ('id', 'state', 'fitness'):
            if want not in cols:
                col, ok = QInputDialog.getItem(None, "Map column",
                                               f"Select column for NODE {want.upper()}:",
                                               cols, 0, False)
                if ok: mapping[col] = want

    # --- launch the worker
    progress = QProgressDialog("Loading network…", None, 0, 0)
    progress.setWindowModality(QtCore.Qt.ApplicationModal)
    progress.setCancelButton(None); progress.show()

    worker = CSVImportWorker(fname, mapping)
    _import_workers.append(worker)

    def on_success(g, fname):
        progress.close()

        # --- 1. make the Edit‑Config window reflect this graph 
        refresh_mutation_configs_from_graph(g)

        # --- 2. swap the new network into all runtime globals 
        global current_graph, pos, initial_graph, initial_pos, step_count,   selected_node, default_total_nodes, default_total_edges

        current_graph        = g
        default_total_nodes  = g.number_of_nodes()     
        default_total_edges  = g.number_of_edges()     

        pos          = nx.spring_layout(current_graph, k=default_spacing)
        initial_graph = copy.deepcopy(current_graph)
        initial_pos   = copy.deepcopy(pos)

        step_count    = 0
        selected_node = None

        # --- 3. redraw everything
        visualize_graph(current_graph, ax, info_ax, pos)
        update_step_label()

        # --- 4. tidy up the worker thread
        worker.quit(); worker.wait()
        _import_workers.remove(worker)

    def on_error(msg):
        progress.close()
        QMessageBox.critical(None, "Import failed", msg)
        worker.quit(); worker.wait()
        _import_workers.remove(worker)

    worker.finished.connect(on_success)
    worker.error.connect(on_error)
    worker.start()


# 3. button on the toolbar
button_import_ax = fig.add_axes([0.01, 0.69, 0.18, 0.08])
button_import = Button(button_import_ax, 'Import Graph')
button_import.on_clicked(import_graph)

def export_graph(event=None):
    """
    Export the current network to a single CSV with six columns:

        source , target , id , state , fitness , color

    • Edge rows   →  source & target filled;  id/state/fitness/color = ''
    • Node rows   →  id/state/fitness/color filled;  source/target   = ''
    """
    fname, _ = QFileDialog.getSaveFileName(
        None, "Export graph configuration", "", "CSV files (*.csv)"
    )
    if not fname:
        return

    # --- ensure .csv extension once 
    fname = str(Path(fname).with_suffix('.csv'))

    # --- node table
    nodes_df = pd.DataFrame(
        [
            (
                int(n),                                              # id
                current_graph.nodes[n].get("state", "healthy"),      # state
                current_graph.nodes[n].get("fitness", 1.0),          # fitness
                STATE_COLORS.get(
                    current_graph.nodes[n].get("state", "healthy"),
                    "#000000"                                        # fallback colour
                )                                                    # color
            )
            for n in current_graph.nodes()
        ],
        columns=["id", "state", "fitness", "color"]
    ).assign(source='', target='')                                   # pad edge cols

    # --- edge list
    edges_df = pd.DataFrame(
        [(int(u), int(v)) for u, v in current_graph.edges()],
        columns=["source", "target"]
    ).assign(id='', state='', fitness='', color='')                  # pad node cols

    # --- concatenate in canonical column order
    combined = pd.concat(
        [
            edges_df[["source", "target", "id", "state", "fitness", "color"]],
            nodes_df[["source", "target", "id", "state", "fitness", "color"]],
        ],
        ignore_index=True
    )

    # --- write to disk
    try:
        combined.to_csv(fname, index=False, na_rep='')               # blanks stay blank
        QMessageBox.information(None, "Export successful", f"Saved:\n• {fname}")
    except Exception as e:
        QMessageBox.critical(None, "Export failed", str(e))


button_export_ax = fig.add_axes([0.46, 0.01, 0.12, 0.09]) # Export Button
button_export = Button(button_export_ax, 'Export configuration')
button_export.on_clicked(export_graph)

current_graph = initialize_graph(default_total_nodes, default_total_edges)
initialize_mutations(current_graph, mutation_configs, healthy_fitness=1)
pos = nx.spring_layout(current_graph, k=default_spacing)
reinit_graph()

visualize_graph(current_graph, ax, info_ax, pos)
update_step_label()

plt.show()