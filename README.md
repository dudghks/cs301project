# CS301-002 Project Milestone 2: Happiness vs. Urban Indicators

This project analyzes the relationship between urban indicators and happpiness score in cities around the world.

The urban indicators used are:

* Average noise level
* Green space area
* Traffic density
* Air quality index
* Healthare index
* Cost of living index

## Group Members
* James Cayetano
* Brandon Zhou
* John Kim
* Aran Kashiani

## Data Sources
Dataset 1: [City Happiness Index - 2024](https://www.kaggle.com/datasets/emirhanai/city-happiness-index-2024)

Dataset 2: [Global Cost of Living](https://www.kaggle.com/datasets/mvieira101/global-cost-of-living)

## Usage
The project is containerized in the `Dockerfile` that executes a python script to generate all visualizations in the notebook. To build it, run the following command in a terminal at the project directory:

```docker build -t cs301-project .```

To generate the output visualizations, run the container with the a mount to the /proj/output folder:

```docker run -v "$(pwd)/output":/proj/output cs301-project```

Alternatively, you may view and execute the notebook using Jupyter Notebook or similar software given you have the dependencies outlined in `requirements.txt` installed. Please note that the Dockerfile does not include the Jupyter Notebook as this was not part of the project specifications.
