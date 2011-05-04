all: transitmodel/gimenez.f90 Fortran/utilities.f90
	f2py -c --opt="-O3 -march=native -ftree-vectorize"  --fcompiler="gnu95" -m gimenez_f transitmodel/gimenez.f90
	f2py -c --opt="-O3 -march=native -ftree-vectorize"  --fcompiler="gnu95" -m utilities_f Fortran/utilities.f90

	mv gimenez_f.so transitmodel/

openmp: transitmodel/gimenez.f90 Fortran/utilities.f90
	f2py -c --opt="-O3 -march=native -fopenmp -ftree-vectorize -ffast-math"  --f90exec="/usr/bin/mpif90" -lgomp -m gimenez_f transitmodel/gimenez.f90
	#f2py -c --opt="-O3 -march=native -fopenmp -ftree-vectorize -ffast-math"  --fcompiler="gnu95" -lgomp -m gimenez_f transitmodel/gimenez.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1

	f2py -c --opt="-O3 -march=native -fopenmp -ftree-vectorize"  --fcompiler="gnu95" -lgomp -m utilities_f Fortran/utilities.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1

	mv gimenez_f.so transitmodel/
