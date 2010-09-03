module Gimenez_f
  use, intrinsic :: ISO_C_BINDING

  implicit none

  integer, parameter :: FS = C_FLOAT
  integer, parameter :: FD = C_DOUBLE

contains
#ifdef __GFORTRAN__
#if __GNUC_MINOR__ < 4
  real(8) pure function log_gamma(x)
    implicit none
    real(8), intent(in) :: x
    log_gamma = log(gamma(x))
  end function log_gamma
#endif
#endif

  subroutine Gimenez(nz, nu, z, r, u, npol, res, nt)
    use omp_lib
    implicit none
    integer, intent(in) :: nz, nu
    real(8), intent(in), dimension(nz) :: z
    real(8), intent(in) :: r
    real(8), intent(in), dimension(nu) :: u
    integer, intent(in) :: npol
    integer, intent(in) :: nt

    real(8), intent(out), dimension(nz) :: res

    integer :: i

    res = 1._fd
    
    !$ if (nt /= 0) call omp_set_num_threads(nt)

    !$omp parallel do shared(nz, nu, z, r, u, npol, res) private(i) schedule(dynamic)
    do i = 1, nz
       if (z(i) < 1._fd+r) then
          res(i) = Gimenez_s(z(i), r, nu, u, npol)
       end if
    end do
    !$omp end parallel do

  contains

    pure real(8) function Gimenez_s(z, r, nu, u, npol)
      implicit none

      real(8), intent(in) :: z, r
      integer, intent(in) :: nu, npol
      real(8), intent(in), dimension(nu) :: u

      real(8), dimension(nu+1) :: a, n, Cn
      real(8) :: b, c, f

      integer :: i

      a  = 0._fd
      Cn = 1._fd
      n = [(i, i=0,nu+1)]

      b = r/(1._fd+r)
      c = z/(1._fd+r)
      !f = 1._fd

      do i = 0, nu
         a(i+1) = alpha(b, c, i, npol)
      end do

      if (nu > 0) then
         Cn(1) = (1._fd - sum(u)) / (1._fd - sum(n(2:) * u / (n(2:)+2._fd)))
         Cn(2:) = u / (1._fd - n(2:) * u / (n(2:)+2._fd))
      end if
      
      Gimenez_s = 1._fd - sum(a*Cn)
    end function Gimenez_s

!!$  subroutine Gimenez(nz, nu, z, r, u, npol, flux)
!!$    implicit none
!!$    
!!$    integer, intent(in) :: nz, nu
!!$    real(8), intent(in), dimension(n)z  :: z
!!$    real(8), intent(in) :: r
!!$    real(8), intent(in), dimension(nu) :: u
!!$    integer, intent(in) :: npol
!!$    real(8), intent(out), dimension(nz) :: flux
!!$
!!$    real(8), dimension(nu+1, n) :: a
!!$    real(8), dimension(nz)   :: c
!!$    real(8), dimension(nu+1) :: n
!!$    real(8), dimension(nz)   :: f
!!$    real(8), dimension(nu+1) :: Cn = 1.
!!$
!!$    real(8), dimension(nz) :: tmp
!!$
!!$    logical, dimension(nz) :: z_mask
!!$    real(8) :: b
!!$
!!$    integer :: i
!!$
!!$    z_mask = z < (1.+r)
!!$
!!$    a = 0.
!!$    b = r/(1.+r)
!!$    c = z/(1.+r)
!!$
!!$    n = [i, i=0,nu+1]
!!$    f = 1.
!!$
!!$    do i = 0,nu+1
!!$       call alpha(b, c, i, npol, tmp)
!!$       a(i+1, :) = tmp
!!$    end do
!!$
!!$    if (nu > 0) then
!!$       Cn[0] = (1. - sum(u)) / (1. - sum(n[1:] * u[:] / (n[1:]+2.)))
!!$       Cn[1:] = u[:] / (1. - n[1:] * u[:] / (n[1:]+2.))
!!$    end if
!!$
!!$    flux = 1. - sum((a*Cn))
!!$
!!$  end subroutine Gimenez


    pure real(8) function alpha(b, c, n, jn)
      implicit none
      real(8), intent(in) :: b, c
      integer, intent(in) :: n, jn

      real(8)  :: norm, sum, vl
      real(8), dimension(jn) :: e, d

      real(8) :: nu, nm
      integer :: i

      nu = (n+2._fd)/2._fd
      norm = b*b * (1._fd - c*c)**(1._fd + nu) / (nu * GAMMA(1._fd + nu))

      call jacobi_Burkardt(jn, 0._fd, 1._fd+nu, 1._fd-2._fd*c*c, d)
      call jacobi_Burkardt(jn,    nu,    1._fd, 1._fd-2._fd*(1._fd-b), e)

      sum = 0._fd
      do i = 0, jn-1
         nm = LOG_GAMMA(nu+i+1.) - LOG_GAMMA(i+2._fd)
         vl = (-1)**i * (2._fd+2._fd*i+nu)*exp(nm)
         nm = LOG_GAMMA(i+1._fd) + LOG_GAMMA(nu+1._fd) - LOG_GAMMA(i+1._fd+nu)

         e(i+1) = e(i+1) * exp(nm)
         vl   = vl * d(i+1) * e(i+1) * e(i+1)
         sum  = sum + vl
      end do

      alpha = norm * sum

    end function alpha

!!$  subroutine alpha(b, c, n, jn, res)
!!$    implicit none
!!$    real(8), dimension(:), intent(in) :: c
!!$    real(8), intent(in) :: b
!!$    integer, intent(in) :: n, jn
!!$    real(8), dimension(size(c)), intent(out) :: res
!!$
!!$    real(8), dimension(size(c)) :: norm, sum, vl
!!$    real(8), dimension(jn, size(c)) :: d
!!$    real(8), dimension(jn) :: e
!!$
!!$    real(8) :: nu, nm
!!$    integer :: i
!!$
!!$    nu = (n+2.)/2.
!!$    norm = b*b * (1. - c*c)**(1. + nu) / (nu * gamma(1. + nu))
!!$
!!$    call jacobi_Burkardt(jn, 0., 1.+nu, 1.-2.*c*c, d)
!!$    call jacobi_Burkardt(jn, nu,    1., 1.-2.*(1.-b), e)
!!$
!!$    sum = 0.
!!$
!!$    do i = 1, jn
!!$       nm = LOG_GAMMA(nu+i+1.) - LOG_GAMMA(i+2.)
!!$       vl = (-1)**i * (2.+2.*i+nu)*exp(nm)
!!$       nm = LOG_GAMMA(i+1.) + LOG_GAMMA(nu+1.) - LOG_GAMMA(i+1.+nu)	
!!$       
!!$       e(i) = e(i) * exp(nm)
!!$       vl   = vl * d(i,:) * e(i) * e(i)
!!$       sum  = sum + vl
!!$    end do
!!$
!!$    res = norm * sum
!!$
!!$  end subroutine alpha

    pure subroutine jacobi_Burkardt(n, alpha, beta, x, cx)
      implicit none
      integer, intent(in) :: n
      real(8), intent(in) :: alpha, beta, x
      real(8), intent(out), dimension(:) :: cx

      real(8) :: ri, c1, c2, c3, c4
      integer :: i

      cx(1) = 1._fd

      if (n>0) then
         cx(2) = ( 1._fd + 0.5_fd * ( alpha + beta ) ) * x + 0.5_fd * ( alpha - beta )

         do i = 2, n-1
            ri = real(i,8)
            c1 = 2._fd * ri * ( ri + alpha + beta ) * ( 2._fd * ri - 2._fd + alpha + beta )
            c2 = ( 2._fd* ri - 1._fd + alpha + beta ) * ( 2._fd * ri  + alpha + beta ) * ( 2._fd* ri - 2._fd + alpha + beta )
            c3 = ( 2._fd* ri - 1._fd + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
            c4 = - 2._fd * ( ri - 1._fd + alpha ) * ( ri - 1._fd + beta )  * ( 2._fd* ri + alpha + beta )

            cx(i+1) = ( ( c3 + c2 * x ) * cx(i) + c4 * cx(i-1) ) / c1
         end do
      end if

    end subroutine jacobi_Burkardt

!!$subroutine jacobi_Burkardt(n, alpha, beta, x, cx)
!!$  implicit none
!!$  integer, intent(in) :: n
!!$  real(8), intent(in) :: alpha, beta
!!$  real(8), dimension(:), intent(in) :: x
!!$  real(8), dimension(:,:), intent(out) :: cx
!!$
!!$  real(8) :: ri, c1, c2, c3, c4
!!$  integer :: i
!!$
!!$  if ((n<0) .or. (alpha<-1.) .or. (beta<-1.)) then
!!$     print *, 'Error in Jacobi_Burkardt: bad parameters.'
!!$     call exit()
!!$  end if
!!$
!!$  cx(1,:) = 1.
!!$
!!$  if (n>0) then
!!$     cx(2,:) = ( 1. + 0.5 * ( alpha + beta ) ) * x + 0.5 * ( alpha - beta )
!!$
!!$     do i = 3, n
!!$        ri = real(i,8)
!!$        c1 = 2. * ri * ( ri + alpha + beta ) * ( 2. * ri - 2. + alpha + beta )
!!$        c2 = ( 2.* ri - 1. + alpha + beta ) * ( 2. * ri  + alpha + beta ) * ( 2.* ri - 2. + alpha + beta )
!!$        c3 = ( 2.* ri - 1. + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
!!$        c4 = - 2. * ( ri - 1. + alpha ) * ( ri - 1. + beta )  * ( 2.* ri + alpha + beta )
!!$
!!$        cx(i,:) = ( ( c3 + c2 * x ) * cx(i-1,:) + c4 * cx(i-2,:) ) / c1
!!$
!!$     end do
!!$  end if
!!$
!!$end subroutine jacobi_Burkardt
  end subroutine Gimenez
end module Gimenez_f
