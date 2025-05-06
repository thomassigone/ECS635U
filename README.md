# ECS635U - Undergraduate Final Year Project
# Simulation of the Multi-Type Moran Process

> *Simulate and visualise how multiple competing mutations spread across complex networks.*

This tool extends the classical two‑type Moran process to an **arbitrary number of mutation types**, providing interactive analytics and a GUI for researchers and students.

**Author :** Thomas Louis Sigone (BSc Computer Science, QMUL - 220640370)  
**Supervisor :** Dr Marc Roth  
**Institution :** School of Electronic Engineering & Computer Science, Queen Mary University of London  
**Date :** 06 May 2025  
**Main file :** `Moran_Process_Simulation_V3.5.py`

## 1  Quick Setup (Unix/macOS & Windows)

### 1.1 Prerequisites

| Requirement       | Version    | Notes                                                                                                             |
| ----------------- | ---------- | ----------------------------------------------------------------------------------------------------------------- |
| **Git**           | 2.30 +     | Clone the repo & manage patches                                                                                   |
| **Conda / Mamba** | 23 +       | [Miniconda](https://docs.conda.io)                                                                                |
| **Python**        | 3.12.\*    | Exact revision is pinned in the environment file                                                                  |
| *Optional*        | Micromamba | Tiny drop‑in for CI or servers                                                                                    |

### 1.2 Clone the project

```bash
# pick a folder of your choice
$ git clone https://github.com/thomassigone/ECS635U.git
$ cd ECS635U
```

### 1.3 Create & activate the Conda environment

> The `environment.yml` shipped with the repo is cross‑platform; Conda automatically selects the correct build for your OS.

#### Unix / macOS (bash/zsh)

```bash
$ conda env create -f environment.yml
$ conda activate ECS635U
```

#### Windows (PowerShell / CMD)

```powershell
:: Open "Anaconda Prompt" *or* run `conda init powershell` once.
PS> mamba env create -f environment.yml
PS> conda activate ECS635U
```

**Troubleshooting**

| Symptom                                                                 | Fix                                                                                                                  |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| *ResolvePackageNotFound* on Windows‑only packages (`vc`, `menuinst`, …) | Regenerate the YAML with `--from-history` or delete those lines                                                      |
| SSL error on corporate proxies                                          | `conda config --set ssl_verify false` or set `REQUESTS_CA_BUNDLE`                                                    |
| PyQt import crash                                                       | Ensure you launched Python from the activated env; on Windows check that `qt.conf` isn’t shadowed by another install |

### 1.4 Run the application

```bash
$ python -u ./src/Moran_Process_Simulation_V3.5.py
```

*Binary‑heavy libs like **NumPy**, **SciPy**, and **PyQt** may build from source; Conda is strongly recommended*.

### 1.5 Use the application
Use the toolbar buttons to edit the graph and start the simulation.
Right-click on a node for context actions; double-click blank space to add a new node; left click to select a node and view its properties.

---

## 2  Project Overview

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

---

## 3  Acknowledgements
This project builds on algorithmic insights from Goldberg, Roth, and Schwarz (2024), "Parameterized Approximation of the Fixation Probability of the Dominant Mutation in the Multi‑Type Moran Process", and the foundational model introduced by Moran (1958), "Random processes in genetics."

I would like to express my deepest gratitude to Dr Marc Roth for his outstanding guidance, expertise, and encouragement throughout this project. His support and insights have been instrumental in shaping my work and in motivating me to reach my highest academic potential. I would also like to thank the teaching staff at the School of Electronic Engineering and Computer Science at QMUL for their continuous support and dedication.

I am profoundly grateful to my grandfather; without his absolute support and trust, rigorous education, and exemplary character, I would not be where I am today. Fulfilling my dream of studying in the UK was only part of the journey - I later understood that my true dream was to make him proud. Thank you for always believing in me and supporting my aspirations. I miss you deeply, Nonno.
To my grandmother, for her boundless sweetness and kindness.
To my parents, for raising me with love, strength, and unwavering belief in my potential.
To my sister, my lighthouse in London, whose presence has brought me clarity and comfort.
To the rest of my family, for all the love and the foundation you've given me.
To my friends for their encouragement, laughter, and support during challenging moments. Your presence has made this journey lighter and more meaningful.

---

## 4  License

MIT © Thomas Louis Sigone, 2024–25.

---

## 5  Citing

If you use this software in academic work please cite the accompanying paper and this repository.
