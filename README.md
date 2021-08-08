Flexknots for EDGES low band data
=================================

Requirements
------------

External:

- python3
- any C compiler
- MPI
- POLYCHORD


PyPi:

- mpi4py
- Cobaya
- numpy
- matplotlib

Optional:

- fgivenx
- anesthetic


Building
--------

```
make
```

Running
-------

```
mpiexec -np $PROC python3 main.py $KNOTS
```

where ```$PROC``` is the number of processes to use and ```$KNOTS``` is the number of knots to use

for example: ```mpiexec -np 8 python3 main.py 6```


Data Availability
-----------------
https://loco.lab.asu.edu/edges/edges-data-release/

Download Figure 1 of Bowman et al. (2018) (figure1_plotdata.csv, 8 kB)


spline.c
--------

Originally from https://people.sc.fsu.edu/~jburkardt/c_src/spline/spline.html


