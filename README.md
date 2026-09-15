# Training a Ms. Pac-Man Agent (Deep Q-Network)

## Overview & Instructions
This repository contains a Deep Q-Network (DQN) agent trained to play Ms. Pac-Man. The agent learns entirely through trial and error by observing the game pixels and receiving in-game score rewards. The notebook is based on [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn).

**How to open and run the notebook:**
1. Open [`Amelia_Roeschel_pacman_dqn.ipynb`](Amelia_Roeschel_pacman_dqn.ipynb) in Google Colab, Jupyter, or VS Code (Python 3.11–3.13 kernel).
2. If using Colab, select a GPU runtime (**Runtime → Change runtime type → T4 GPU**).
3. The hyperparameters are pre-set in Section 1. Select **Run All** to execute the notebook from top to bottom. Setup, baseline evaluation, training, final evaluation, and the results ZIP download run automatically.

The notebook in this repository is saved with all outputs from the final run, so the scores, plots, and gameplay can be inspected without rerunning it.

---

## ⚙️ Hyperparameters
I used the following settings for this experiment (all other settings were left at the notebook defaults, listed in [config.json](results/config.json)):
* **Exploration Rate:** `0.25` - I chose 25% to allow the agent to rely on its learned policy most of the time while still leaving enough room (1/4 of its moves) to discover new, potentially better paths.
* **Episodes:** `100` - My first run used 5 episodes as a setup check. Once the pipeline worked, I moved to the notebook's 100-episode starting value to give the network enough games to learn from while still finishing in a single Colab session.
* **Learning Rate:** `0.0002` - I used a small learning rate so the network updates its weights gradually, preventing it from wildly forgetting past lessons, while going slightly above the 0.0001 reference point to learn faster within a 100-game budget.

---

## 🧠 How the Agent Works
* **Observations:** The agent "sees" the game as a stack of **four consecutive 84×84 grayscale game screens**. Stacking four frames lets the network perceive movement and direction (like which way the ghosts are moving).
* **Actions:** The agent chooses one of **9 joystick moves**: NOOP (no move), UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT. Each decision is held for four game frames.
* **Rewards:** The reward is the **game points** it earns (eating dots, fruit, or vulnerable ghosts). During learning, each reward is clipped to between −1 and +1, so a 10-point dot and a 200-point ghost count the same to the network; all scores reported below are the real game scores.

---

## 📊 Evaluation & Scores

**Expectations vs. Observations:**
Before training, I expected 100 episodes to produce a clear but modest improvement over the untrained network: moving toward dots more deliberately, while still dying to ghosts often.

After training, the mean evaluation score rose only slightly, from **492 to 534 (+42, about +9%)**. Three of the five games improved and two got worse (seed 202 fell from 500 to 150). With only five games, this gain is small enough that it could be noise rather than real learning.

The training dashboard supports that caution. The 25-game average training score peaked around 830 near episode 20, drifted down to about 510 by episode 83, and ended near 660. The training loss rose steadily from about 0.02 to 0.07–0.08 instead of falling.

In the gameplay, the trained agent's best game (seed 505, 920 points in total) scored 210 points in its first ~10 seconds, then stopped collecting points for the rest of the 20-second clip and lost a life. For seed 101, the first 20 seconds of play after 100 episodes are frame-for-frame identical to the untrained agent, even though the full-game score rose from 350 to 630. Whatever the agent learned shows up later in the game, not in its opening moves.

### Score Comparison
Evaluation used the notebook's fixed settings before and after training: the same five seeds, 5% exploration, and a 3,000-decision time limit. The baseline is the untrained network. Full data: [comparison.json](results/comparison.json).

| Evaluation Game | Baseline (Untrained) Score | Trained Score |
| :--- | :--- | :--- |
| **Seed 101** | 350 | 630 |
| **Seed 202** | 500 | 150 |
| **Seed 303** | 320 | 420 |
| **Seed 404** | 800 | 550 |
| **Seed 505** | 490 | 920 |
| **MEAN SCORE** | **492.0** | **534.0** |

No evaluation game reached the time limit, before or after training. Every game ended at game over.

---

## 📈 Training Statistics & Hardware
* **Run status:** Completed (not interrupted). Run ID `20260915_224059_130181`.
* **Hardware Used:** NVIDIA T4 GPU (CUDA) via Google Colab, Python 3.13.15, PyTorch 2.11.0
* **Completed Episodes:** 100 of 100
* **Total Decisions (Steps):** 59,821
* **Total Learning Updates:** 14,706
* **Elapsed Training Time:** 225 seconds (about 3.8 minutes, including the periodic gameplay samples)

Source: [training_summary.json](results/training_summary.json)

An earlier 5-episode run was used only as a setup check and is not the submitted result.

---

## 🔬 Limitations & Next Steps
* **Observed Limitation:** Training did not steadily improve play. Training scores drifted down while the loss rose, and in the gameplay the agent stalls after its first burst of points instead of moving toward the remaining dots. One likely cause: the agent only remembers its last 5,000 decisions (about 8 games) and keeps making 25% random moves, so it keeps relearning from a small, noisy set of recent experience instead of building on everything it has seen.
* **Next Experiment:** I would change only the **Learning Rate**, from `0.0002` to `0.0001`, and keep exploration at 0.25 and episodes at 100. The rising loss and falling training scores suggest each update was too large for such a small memory. Halving the step size should make learning steadier, which I would check by looking for a flatter loss curve and a training score that trends up instead of down.

---

## 📁 Visual Evidence & Artifacts
GIFs show the first 20 seconds of a game at 4× speed.

### Untrained Baseline Gameplay
First evaluation game (seed 101), before any training:

![Untrained Agent](results/untrained.gif)

### Best Trained Gameplay
Highest-scoring of the five final evaluation games (seed 505, 920 points in total):

![Best Trained Agent](results/best_trained.gif)

### Intermediate Gameplay (every 25 episodes)
Each sample plays seed 101 with the network at that point in training.

| After 25 episodes (score 200) | After 50 episodes (score 250) |
| :---: | :---: |
| ![Episode 25](results/episode_0025.gif) | ![Episode 50](results/episode_0050.gif) |
| **After 75 episodes (score 640)** | **After 100 episodes (score 630)** |
| ![Episode 75](results/episode_0075.gif) | ![Episode 100](results/episode_0100.gif) |

### Training Dashboard
![Training Dashboard](results/training_dashboard.png)

### Raw Data Links
* [Executed Notebook (Amelia_Roeschel_pacman_dqn.ipynb)](Amelia_Roeschel_pacman_dqn.ipynb)
* [comparison.json](results/comparison.json)
* [config.json](results/config.json)
* [training.csv](results/training.csv)
* [training_summary.json](results/training_summary.json)

### Model Checkpoints
Model checkpoints (`untrained.pt`, `episode_0025.pt` through `episode_0100.pt`, and `trained.pt`, about 6 MB each) are not stored in this repository to keep it small. They are saved in the full results ZIP from this run (`20260915_224059_130181.zip`), which I keep locally.
