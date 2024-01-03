# MHC-binding

Predicting MHC binding affinity using deep learning. This repository was created as part of the recruitment process for an internship position at Insta Deep. The deadline was short so it is more focused on the method than the results.

## Set up

The code was written in Python 3.11.4. Poetry was used to manage the dependencies. And a makefile was used to create shortcuts for the most common commands.

To install the project:

```bash
make install
```

Then you need to put the data in the `data` folder at the root of the project.

## Usage

Data visualization and analysis is done in the notebook `data_exploration.ipynb`.
The models are trained and analysed in the notebook `main.ipynb`.
