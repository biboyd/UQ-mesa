#!/bin/bash
python run_quadratic_analysis.py output2.csv -lo 0.01 0.3 -hi 0.1 0.9 -v -nm 1000 -o run_quad_analysis.log
python run_quadratic_analysis.py output2.csv -lo 0.01 0.3 -hi 0.1 0.9 -v -nens 6 --maxensemble 500 -o run_quad_ensemble-6.dat


