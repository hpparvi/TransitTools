!!=== Gimenez transit shape ===
!!
!! This module implements the transit shape model by
!! Alvaro Gimenez (A&A 450, 1231--1237, 2006).
!!
!! The code is adapted from the Fortran implementation
!! at http://thor.ieec.uab.es/LRVCode. The major changes
!! are in the parallelization and vectorization. 
!!
!! Instead of repeating the computations for each lightcurve point
!! separately, we minimize the redundant calculations by
!! computing the common factors for all lightcurve points.
!! This can give us a speed up of several orders of magnitude. 
!!
!! -GPL-
!! This program is free software: you can redistribute it and/or modify
!! it under the terms of the GNU General Public License as published by
!! the Free Software Foundation, either version 3 of the License, or
!! (at your option) any later version.
!!
!! This program is distributed in the hope that it will be useful,
!! but WITHOUT ANY WARRANTY; without even the implied warranty of
!! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!! GNU General Public License for more details.
!!
!! You should have received a copy of the GNU General Public License
!! along with this program.  If not, see <http://www.gnu.org/licenses/>.
!! -GPL-
!!
!! Author
!!  Hannu Parviainen <hannu@iac.es>
!!
!! Date 
!!  01.03.2011
!!
module gimenez
  use, intrinsic :: ISO_C_BINDING
  implicit none

  include "mpif.h"

  integer, parameter :: FS = C_FLOAT
  integer, parameter :: FD = C_DOUBLE

  real(8), allocatable, dimension(:,:) :: a_e_nm1, a_e_nm2, a_e_vl1
  real(8), allocatable, dimension(:,:,:) :: j_d, j_e
  real(8), allocatable, dimension(:) :: tmp1d1, tmp1d2
  integer :: init_n = 0

  procedure (), pointer :: p

contains
  subroutine init(npol, nldc)
    implicit none
    integer, intent(in) :: npol, nldc
    integer :: i, j
    real(8) :: nu

    if (allocated(a_e_nm1)) deallocate(a_e_nm1)
    if (allocated(a_e_nm2)) deallocate(a_e_nm2)
    if (allocated(a_e_vl1)) deallocate(a_e_vl1)
    if (allocated(j_d)) deallocate(j_d)
    if (allocated(j_e)) deallocate(j_e)
    if (allocated(tmp1d1)) deallocate(tmp1d1)
    if (allocated(tmp1d2)) deallocate(tmp1d2)

    allocate(a_e_nm1(npol, nldc+1), a_e_nm2(npol, nldc+1), a_e_vl1(npol, nldc+1))
    allocate(j_d(4, npol, nldc+1), j_e(4, npol, nldc+1))
    allocate(tmp1d1(1000000), tmp1d2(1000000))

    do j=1,nldc+1
       nu = (real(j-1,8)+2._fd)/2._fd
       do i = 0, npol-1
          a_e_nm1(i+1, j) = exp(log_gamma(nu+i+1._fd) - log_gamma(i+2._fd))
          a_e_nm2(i+1, j) = exp(log_gamma(i+1._fd) + log_gamma(nu+1._fd) - log_gamma(i+1._fd+nu))
          a_e_vl1(i+1, j) = (-1)**(i) * (2._fd+2._fd*i+nu) * a_e_nm1(i+1, j)
       end do
    end do

    !! Jacobi coefficients
    !!
    do j=1,nldc+1
       nu = (real(j-1,8)+2._fd)/2._fd
       do i = 0, npol-1
          call j_coeff(0._fd, 1._fd+nu, real(i+1,8), j_d(:,i+1,j))
          call j_coeff(   nu,    1._fd, real(i+1,8), j_e(:,i+1,j))
       end do
    end do
  
  contains
    subroutine j_coeff(alpha, beta, ri, res)
      implicit none
      real(8), intent(in) :: alpha, beta, ri
      real(8), dimension(4), intent(inout):: res
      res(1) = 2._fd * ri * ( ri + alpha + beta ) * ( 2._fd * ri - 2._fd + alpha + beta )
      res(2) = ( 2._fd* ri - 1._fd + alpha + beta ) * ( 2._fd * ri  + alpha + beta ) * ( 2._fd* ri - 2._fd + alpha + beta )
      res(3) = ( 2._fd* ri - 1._fd + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
      res(4) = - 2._fd * ( ri - 1._fd + alpha ) * ( ri - 1._fd + beta )  * ( 2._fd* ri + alpha + beta )
    end subroutine j_coeff
  end subroutine init

  subroutine finalize()
    implicit none
    if (allocated(a_e_nm1)) deallocate(a_e_nm1)
    if (allocated(a_e_nm2)) deallocate(a_e_nm2)
    if (allocated(a_e_vl1)) deallocate(a_e_vl1)
    if (allocated(j_d)) deallocate(j_d)
    if (allocated(j_e)) deallocate(j_e)
  end subroutine finalize

  subroutine eval_t(t, r, u, npol, t0, p, a, i, nthreads, nt, nu, res)
    use omp_lib
    implicit none
    integer, intent(in) :: nt, nu, nthreads, npol
    real(8), intent(in), dimension(nt) :: t
    real(8), intent(in), dimension(nu) :: u
    real(8), intent(in) :: r, t0, p, a, i
    real(8), intent(out), dimension(nt) :: res

    real(8) :: p_inv
    integer :: j

    p_inv = 1._fd/p
    !$omp parallel do private(i) shared(tmp1d1, tmp1d2, t, t0, p_inv)
    do j=1,nt
       tmp1d1(j) = (t(j)-t0)*p_inv                                          !phase -1 -- 1
       tmp1d2(j) = sign(1._fd, -(mod(tmp1d1(j) + 0.25_fd, 1._fd) - 0.5_fd)) !sign
       tmp1d1(j) = 6.28318*tmp1d1(j)                                        !phase -2pi -- 2pi
       tmp1d1(j) = tmp1d2(j)*a*sqrt(sin(tmp1d1(j))**2 + (cos(i)*cos(tmp1d1(j)))**2) !projected distance z
    end do
    !$omp end parallel do
    call eval(tmp1d1(1:nt), r, u, npol, 1._fd, nthreads, nt, nu, res)
  end subroutine eval_t

  subroutine eval(z, r, u, npol, zeropoint, nt, nz, nu, res)
    use omp_lib
    implicit none
    integer, intent(in) :: nz, nu, nt, npol
    real(8), intent(in), dimension(nz) :: z
    real(8), intent(in), dimension(nu) :: u
    real(8), intent(in) :: r, zeropoint
    real(8), intent(out), dimension(nz) :: res
    
    logical, dimension(nz) :: mask
    real(8), dimension(:), allocatable :: z_t, res_t
    integer :: nz_t
    real(8) :: t_start, t_finish

    !t_start = mpi_wtime()
    ! $ if (nt /= 0) call omp_set_num_threads(nt)

    res  = 0._fd
    mask = (z > 0._fd) .and. (z < 1._fd+r)
    nz_t = count(mask)

    allocate(z_t(nz_t), res_t(nz_t))
    z_t   = pack(z, mask)
    res_t = gimenez_v(z_t, u, r, npol)
    res   = unpack(res_t, mask, res)
    deallocate(z_t, res_t)
    
    res = zeropoint + res
    !t_finish = mpi_wtime()
    !print *, t_finish - t_start
  end subroutine eval


  function gimenez_v(z, u, r, npol)
    implicit none
    real(8), intent(in), dimension(:) :: z
    real(8), intent(in), dimension(:) :: u    
    real(8), intent(in) :: r
    integer, intent(in) :: npol

    real(8), dimension(size(z)) :: gimenez_v

    real(8), dimension(size(z), size(u)+1) :: a 
    real(8), dimension(size(u)+1) :: n, Cn
    real(8), dimension(1) :: b
    real(8), dimension(size(z)) :: c !,b
    integer :: i, j 

    a  = 0._fd
    Cn = 1._fd
    n  = [(i, i=0,size(u))]
    b  = r/(1._fd+r)
    c  = z/(1._fd+r)
    
    do j=1,size(u)+1
       call alpha(b, c, j-1, npol, a(:,j))
    end do

    if (size(u) > 0) then
       Cn(1) = (1._fd - sum(u)) / (1._fd - sum(u*n(2:) / (n(2:)+2._fd)))
       Cn(2:) = u / (1._fd - n(2:) * u / (n(2:)+2._fd))
    end if
    
    do i=1,size(z)
       gimenez_v(i) = -sum(a(i,:)*Cn)
    end do
  end function gimenez_v
  
  !!--- Alpha ---
  !!
  subroutine alpha(b, c, n, npol, a)
    implicit none
    real(8), intent(in), dimension(:) :: b, c
    integer, intent(in) :: n, npol
    real(8), dimension(size(c)), intent(out) :: a

    real(8), dimension(size(c)) :: norm, vl, sm
    real(8), dimension(size(b), npol) :: e
    real(8), dimension(size(c), npol) :: d

    real(8) :: nu
    integer :: i

    nu   = (real(n,8)+2._fd)/2._fd

    !$omp parallel workshare default(shared)
    norm = b(1)*b(1) * (1._fd - c*c)**(1._fd + nu) / (nu * GAMMA(1._fd + nu))
    !$omp end parallel workshare

    call jacobi(npol, 0._fd, 1._fd+nu, 1._fd-2._fd*c**2,      n+1, j_d, d)
    call jacobi(npol,    nu,    1._fd, 1._fd-2._fd*(1._fd-b), n+1, j_e, e)
    
    sm = 0._fd
    !$omp parallel default(none) private(i,vl) shared(a,a_e_nm1,a_e_nm2,a_e_vl1,npol,nu,b,c,d,e,sm,norm,n)
    !$omp do &
    !$omp schedule(static) &
    !$omp reduction(+:sm)
    do i = 1, npol
       e(1,i) = e(1,i) * a_e_nm2(i,n+1)
       sm  = sm + (a_e_vl1(i,n+1) * d(:,i) * e(1,i)**2)
    end do
    !$omp end do
       
    !$omp workshare
    a = norm * sm
    !$omp end workshare

    !$omp end parallel
  end subroutine alpha

  !!--- Jacobi polynomials ---
  !!
  !! Adapted from the Jacobi polynomials routine by J. Burkardt. The
  !! only major difference is that the routine computes the values for
  !! multiple x at the same time.
  !!
  subroutine jacobi(npol, alpha, beta, x, i_ld, j_c, cx)
    use omp_lib
    implicit none
    integer, intent(in) :: npol, i_ld
    real(8), intent(in) :: alpha, beta
    real(8), intent(in), dimension(:) :: x
    real(8), intent(in), dimension(:,:,:) :: j_c
    real(8), intent(out), dimension(:,:) :: cx

    integer :: i, j
    j = i_ld

    !$omp parallel default(shared)
    !$omp workshare 
    cx(:,1) = 1._fd
    cx(:,2) = ( 1._fd + 0.5_fd * ( alpha + beta ) ) * x + 0.5_fd * ( alpha - beta )
    !$omp end workshare
    do i = 2, npol-1
       !$omp workshare
       cx(:, i+1) = ( ( j_c(3,i,j) + j_c(2,i,j) * x ) * cx(:,i) + j_c(4,i,j) * cx(:,i-1) ) / j_c(1,i,j)
       !$omp end workshare
    end do
    !$omp end parallel
  end subroutine jacobi

end module gimenez
