
# NFL Big Data Bowl 2025 - Separation Consistency Index (SCI)

This repository contains code for analyzing NFL player tracking data from the Big Data Bowl 2025 competition, with a focus on the **Separation Consistency Index (SCI)**, a novel metric for evaluating how consistently receivers create and maintain separation during routes.

## Overview

The Separation Consistency Index (SCI) measures how reliably a receiver can create and maintain separation from defenders throughout a route. While traditional metrics like average separation only capture the typical distance to defenders, SCI incorporates consistency, which can reveal important differences in receiver performance.

Key advantages of SCI:
- Identifies receivers who maintain steady separation vs. those who alternate between wide open and tightly covered
- Correlates with actual reception probability
- Leverages the granular tracking data to extract meaningful route-running insights
- Provides a new dimension for evaluating receiver performance beyond traditional stats

## Repository Structure



## Installation

This project requires Python 3.13 or later.

1. Clone the repository:
```
git clone https://github.com/yourusername/nfl_big_data_bowl_2025.git
cd nfl_big_data_bowl_2025
```

2. Set up a virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```
pip install -e .
```

## Dataset

The NFL Big Data Bowl 2025 dataset contains:
- `games.csv` - Game-level information (gameId, teams, date, score, etc.)
- `plays.csv` - Play-level information (gameId, playId, down, yardsToGo, formation, etc.)
- `players.csv` - Player metadata (nflId, name, height, weight, position, etc.)
- `player_play.csv` - Player participation and stats per play
- `tracking_week[X].csv` - Player tracking data at ~10 frames/second (for weeks 1-9)

To download the dataset, you'll need Kaggle credentials and can use:
```
python main.py download
```

## Usage

### Command-line Interface

The repository provides a command-line interface for common tasks:

1. Download the dataset:
```
python main.py download [--force]
```

2. Analyze a specific play:
```
python main.py analyze-play [--week WEEK] [--game-id GAME_ID] [--play-id PLAY_ID] [--random]
```

3. Run SCI analysis for an entire week:
```
python main.py analyze-week [--week WEEK] [--max-plays MAX_PLAYS]
```

### Jupyter Notebook

For interactive exploration, use the provided notebook:
```
cd src/notebooks
jupyter notebook sci_analysis_example.ipynb
```

## Separation Consistency Index Methodology

SCI is calculated using the following steps:

1. For each pass play, identify the route-running phase (from snap to pass)
2. For each frame during the route, compute the minimum distance between the receiver and any defender
3. Calculate metrics on this distance time series:
   - Mean separation: Average distance to nearest defender
   - Standard deviation: Variation in separation distance
   - Percentage of time "open": Frames with separation ≥ 3 yards
   - SCI score: Ratio of mean separation to standard deviation

A higher SCI score indicates a receiver who maintains more consistent separation throughout routes. The final player-level metrics aggregate these values across multiple plays.

## Examples

### Single Play Analysis
![Single Play Analysis](https://example.com/single_play.png)

The single play analysis shows the separation distance over time for a specific receiver during a route, along with the route path visualization.

### Player Comparison
![Player Comparison](https://example.com/player_comparison.png)

The player comparison view allows comparing multiple receivers across different SCI metrics.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT License - see the LICENSE file for details.

## Acknowledgments

- NFL for providing the tracking data through the Big Data Bowl competition
- Kaggle for hosting the competition
