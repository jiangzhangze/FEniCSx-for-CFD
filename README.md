# FEniCSx-for-CFD
My codes for computational fluid dynamics ：writed by gmsh‘s python API.
This reporsitory is mainly about flow around cylinder.

## Requirement
```
FEniCSx >= 0.5.0
gmsh
```

## Install gmsh 
`pip install --upgrade gmsh`

## Install dolfinx
[Anaconda](https://www.anaconda.com/) is recommended for installing FEniCSx.
```
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```
