OPTS = "-O3 -march=native -fopenmp -ftree-vectorize -ffast-math"
FLAGS = -DF2PY_REPORT_ON_ARRAY_COPY=1
CMP = --f90exec=/usr/bin/mpif90
#CMP = --fcompiler="gnu95"

all: transitmodel/gimenez.f90 Fortran/utilities.f90
	f2py -c --opt="-O3 -march=native -ftree-vectorize"  --fcompiler="gnu95" -m gimenez_f transitmodel/gimenez.f90
	f2py -c --opt="-O3 -march=native -ftree-vectorize"  --fcompiler="gnu95" -m utilities_f Fortran/utilities.f90

	mv gimenez_f.so transitmodel/

openmp: transitmodel/gimenez.f90 Fortran/utilities.f90 fitting/fitnessfunction.f90
	f2py -c --opt=${OPTS} ${CMP} -lgomp -m gimenez_f transitmodel/gimenez.f90 ${FLAGS}
	f2py -c --opt=${OPTS} ${CMP} -lgomp -m fitnessfunction_f fitting/fitnessfunction.f90 ${FLAGS}
	f2py -c --opt=${OPTS} ${CMP} -lgomp -m utilities_f Fortran/utilities.f90 ${FLAGS}

	mv gimenez_f.so transitmodel/
	mv fitnessfunction_f.so fitting/