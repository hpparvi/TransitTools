all: Fortran/Gimenez.f90 Fortran/utilities.f90
	f2py -c --opt="-O3 -march=native -fopenmp -ftree-vectorize"  --fcompiler="gnu95" -lgomp -m Gimenez_f Fortran/Gimenez.f90
	f2py -c --opt="-O3 -march=native -fopenmp -ftree-vectorize"  --fcompiler="gnu95" -lgomp -m utilities_f Fortran/utilities.f90
