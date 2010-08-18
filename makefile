
home: Gimenez.f90
	f2py -c --opt="-O3 -march=barcelona -fopenmp"  --f90exec="gfortran" -lgomp -m Gimenez_f Gimenez.f90

iac: Gimenez.f90
	f2py -c --opt="-O3 -march=barcelona -ftree-vectorize -fopenmp"  --fcompiler="gfortran" -lgomp -m Gimenez_f Gimenez.f90

iac_intel:
	f2py -c --opt="-fast"  --fcompiler="intelem" -m Gimenez_f Gimenez.f90

#	f2py -c --opt="-O3 -m32"  --fcompiler="pg" --arch='-m32' --compiler="unix" -m futils futils.f95
#	f2py -c --opt="-O3 -march=barcelona -m32"  --fcompiler="g95" -lgomp -m futils futils.f95
#	f2py -c --opt="-O3 -march=barcelona -fopenmp"  --fcompiler="gnu95" --f90exec='/usr/bin/f95' -lgomp -m futils futils.f95
#	f2py -c --opt="-O3 -march=barcelona -fopenmp"  --fcompiler="gfortran" -lgomp -m futils futils.f95