OPTS = "-O3 -march=native -ftree-vectorize -ffast-math"
OPTS_OMP = "-O3 -march=native -fopenmp -ftree-vectorize -ffast-math -ftree-vectorizer-verbose=2"
LIBS = -lacml_mp -lacml_mv

FLAGS = -DF2PY_REPORT_ON_ARRAY_COPY=1
#CMP = --f90exec=/usr/bin/mpif90
CMP = --f90exec=//opt/openmpi/bin/mpif90
#CMP = --fcompiler="gnu95"

all: transitmodel/gimenez.f90 Fortran/utilities.f90 fitting/fitnessfunction.f90
	f2py -c --opt=${OPTS} ${CMP} -L${ACML_PATH} ${LIBS} -m gimenez_f transitmodel/gimenez.f90 ${FLAGS}
	f2py -c --opt=${OPTS} ${CMP} -L${ACML_PATH} ${LIBS} -m fitnessfunction_f fitting/fitnessfunction.f90 ${FLAGS}
	f2py -c --opt=${OPTS} ${CMP} -L${ACML_PATH} ${LIBS} -m utilities_f Fortran/utilities.f90 ${FLAGS}

	mv gimenez_f.so transitmodel/
	mv fitnessfunction_f.so fitting/

openmp: transitmodel/gimenez.f90 Fortran/utilities.f90 fitting/fitnessfunction.f90
	f2py -c --opt=${OPTS_OMP} ${CMP} -L${ACML_PATH} ${LIBS} -lgomp -m gimenez_f transitmodel/gimenez.f90 ${FLAGS}
	f2py -c --opt=${OPTS_OMP} ${CMP} -L${ACML_PATH} ${LIBS} -lgomp -m fitnessfunction_f fitting/fitnessfunction.f90 ${FLAGS}
	f2py -c --opt=${OPTS_OMP} ${CMP} -L${ACML_PATH} ${LIBS} -lgomp -m utilities_f Fortran/utilities.f90 ${FLAGS}

	mv gimenez_f.so transitmodel/
	mv fitnessfunction_f.so fitting/