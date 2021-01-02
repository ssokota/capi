# capi

This repository implements CAPI for the game Trade Comm. 
CAPI is an approximate policy iteration algorithm that operates within the PuB-MDP of common-payoff games. 
It was introduced in the paper and thesis _Solving Common-Payoff Games with Approximate Policy Iteration_.

## Installation
- Clone the repository using the command `git clone https://github.com/ssokota/capi.git`
- Enter the directory using `cd capi`
- Install the package with the command `pip install .`

## Reproducing Results
The command `python scripts/interface.py` executes one run with the same settings as those used to generate results from Figure 4 of the AAAI paper.
One run takes a few hours on a GPU.
Try the command `python scripts/interface.py --num_items 5 --num_utterances 5` for computationally cheaper results on a smaller version of Trade Comm.

## References

For an introduction to CAPI, see either the AAAI paper or the thesis below.
```
@inproceedings{capi_paper, 
title       = {Solving Common-Payoff Games with Approximate Policy Iteration}, 
journal     = {Proceedings of the AAAI Conference on Artificial Intelligence}, 
author      = {Sokota, Samuel and Lockhart, Edward and Timbers, Finbarr and Davoodi, Elnaz and D’Orazio, Ryan and Burch, Neil and Schmid, Martin and Bowling, Michael and Lanctot, Marc}, 
year        = {2021}
}
```
```
@mastersthesis{capi_thesis,
author       = {Samuel Sokota}, 
title        = {Solving Common-Payoff Games with Approximate Policy Iteration},
school       = {University of Alberta},
year         = {2020},
}
```

For a Dec-POMDP implementation of Trade Comm, see OpenSpiel.
```
@misc{openspiel,
title   = {OpenSpiel: A Framework for Reinforcement Learning in Games}, 
author  = {Marc Lanctot and Edward Lockhart and Jean-Baptiste Lespiau and Vinicius Zambaldi and Satyaki Upadhyay and Julien Pérolat and Sriram Srinivasan and Finbarr Timbers and Karl Tuyls and Shayegan Omidshafiei and Daniel Hennes and Dustin Morrill and Paul Muller and Timo Ewalds and Ryan Faulkner and János Kramár and Bart De Vylder and Brennan Saeta and James Bradbury and David Ding and Sebastian Borgeaud and Matthew Lai and Julian Schrittwieser and Thomas Anthony and Edward Hughes and Ivo Danihelka and Jonah Ryan-Davis},
year    = {2020}
}
```
