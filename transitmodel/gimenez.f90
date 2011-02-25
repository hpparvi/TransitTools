!!=== Gimenez transit shape ===
!!
!! This module implements the transit shape model by
!! Alvaro Gimenez (A&A 450, 1231--1237, 2006).
!!
!! The code is adapted from the Fortran implementation
!! at http://thor.ieec.uab.es/LRVCode. The major changes
!! are in the parallelization and vectorization. 
!!
!! Instead
!! of repeating the computations for each lightcurve point
!! separately, we minimize the redundant calculations by
!! computing the common factors for all lightcurve points.
!! This can give us a speed up of several orders of magnitude. 
!!
!! Author
!!  Hannu Parviainen
!!
module gimenez
  use, intrinsic :: ISO_C_BINDING
  implicit none

  integer, parameter :: FS = C_FLOAT
  integer, parameter :: FD = C_DOUBLE

contains
  subroutine c_gimenez(z, r, u, npol, nz, nu, nt, res)
    use omp_lib
    implicit none
    integer, intent(in) :: nz, nu, nt, npol
    real(8), intent(in), dimension(nz) :: z
    real(8), intent(in), dimension(nu) :: u
    real(8), intent(in) :: r
    real(8), intent(out), dimension(nz) :: res

    logical, dimension(nz) :: tmask
    real(8), dimension(nz) :: z_tr, tres

    !$ if (nt /= 0) call omp_set_num_threads(nt)

    res   = 1._fd
    tmask = (z > 0._fd) .and. (z < 1._fd+r)
    z_tr  = pack(z, tmask)
    tres  = gimenez_v(z_tr, u, r, npol)
    res   = unpack(tres, tmask, res)

  end subroutine c_gimenez


  function gimenez_v(z, u, r, npol)
    implicit none
    real(8), intent(in), dimension(:) :: z
    real(8), intent(in), dimension(:) :: u    
    real(8), intent(in) :: r
    integer, intent(in) :: npol

    real(8), dimension(size(z)) :: gimenez_v

    real(8), dimension(size(z), size(u)+1) :: a 
    real(8), dimension(size(u)+1) :: n, Cn    
    real(8), dimension(size(z)) :: b, c
    integer :: i, j

    a  = 0._fd
    Cn = 1._fd
    n  = [(i, i=0,size(u)+1)]
    b  = r/(1._fd+r)
    c  = z/(1._fd+r)
    
    do j=1,size(u)+1
       a(:,j) = alpha(b, c, j-1, npol)
    end do

    if (size(u) > 0) then
       Cn(1) = (1._fd - sum(u)) / (1._fd - sum(u*n(2:) / (n(2:)+2._fd)))
       Cn(2:) = u / (1._fd - n(2:) * u / (n(2:)+2._fd))
    end if
    
    do i=1,size(z)
       gimenez_v(i) = 1._fd - sum(a(i,:)*Cn)
    end do
  end function gimenez_v
  
  !!--- Alpha ---
  !!
  !!
  !!TODO: Parallelize this with OpenMP 
  function alpha(b, c, n, npol)
    implicit none
    real(8), intent(in), dimension(:) :: b, c
    integer, intent(in) :: n, npol
    real(8), dimension(size(c)) :: alpha

    real(8), dimension(size(c)) :: norm, vl, sm
    real(8), dimension(size(c), npol) :: e, d

    real(8) :: nu, nm
    integer :: i

    nu   = (real(n,8)+2._fd)/2._fd
    norm = b*b * (1._fd - c*c)**(1._fd + nu) / (nu * GAMMA(1._fd + nu))

    call jacobi(npol, 0._fd, 1._fd+nu, 1._fd-2._fd*c**2,      d)
    call jacobi(npol,    nu,    1._fd, 1._fd-2._fd*(1._fd-b), e)

    sm = 0._fd
    do i = 0, npol-1
       nm = log_gamma(nu+i+1.) - log_gamma(i+2._fd)
       vl = (-1)**i * (2._fd+2._fd*i+nu)*exp(nm)
       nm = log_gamma(i+1._fd) + log_gamma(nu+1._fd) - log_gamma(i+1._fd+nu)
       e(:, i+1) = e(:, i+1) * exp(nm)
       sm  = sm + (vl * d(:,i+1) * e(:,i+1)**2) 
    end do

    alpha = norm * sm
  end function alpha

  !!--- Jacobi polynomials ---
  !!
  !! Adapted from the Jacobi polynomials routine by J. Burkardt. The
  !! only major difference is that the routine computes the values for
  !! multiple x at the same time.
  !!
  pure subroutine jacobi(npol, alpha, beta, x, cx)
    implicit none
    integer, intent(in) :: npol
    real(8), intent(in) :: alpha, beta
    real(8), intent(in), dimension(:) :: x
    real(8), intent(out), dimension(:,:) :: cx

    real(8) :: ri, c1, c2, c3, c4
    integer :: i

    cx(:,1) = 1._fd
    cx(:,2) = ( 1._fd + 0.5_fd * ( alpha + beta ) ) * x + 0.5_fd * ( alpha - beta )
    do i = 2, npol-1
       ri = real(i,8)
       c1 = 2._fd * ri * ( ri + alpha + beta ) * ( 2._fd * ri - 2._fd + alpha + beta )
       c2 = ( 2._fd* ri - 1._fd + alpha + beta ) * ( 2._fd * ri  + alpha + beta ) * ( 2._fd* ri - 2._fd + alpha + beta )
       c3 = ( 2._fd* ri - 1._fd + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
       c4 = - 2._fd * ( ri - 1._fd + alpha ) * ( ri - 1._fd + beta )  * ( 2._fd* ri + alpha + beta )
       cx(:, i+1) = ( ( c3 + c2 * x ) * cx(:,i) + c4 * cx(:,i-1) ) / c1
    end do
  end subroutine jacobi

end module gimenez
