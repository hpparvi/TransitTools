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

  subroutine Gimenez(z, r, u, npol, nz, nu, nt, res, supersampling)
    use omp_lib
    implicit none
    integer, intent(in) :: nz, nu, nt, npol
    real(8), intent(in), dimension(nz) :: z
    real(8), intent(in), dimension(nu) :: u
    real(8), intent(in) :: r
    real(8), intent(out), dimension(nz) :: res
    integer, intent(in), optional :: supersampling

    integer :: i, j, ss
    real(8) :: bw, hbw, ss_norm

    res = 1._fd

    !$ if (nt /= 0) call omp_set_num_threads(nt)

    if(present(supersampling)) then
       ss = supersampling
    else
       ss = 1
    end if

    !!--- Supersampling ---
    !!
    if (ss > 1) then
       ss_norm = 1._fd / real(ss, fd)
       !$omp parallel do &
       !$omp shared(nz, nu, z, r, u, npol, res, ss, ss_norm) &
       !$omp private(i, j, bw, hbw) &
       !$omp schedule(dynamic)
       do i = 1, nz
          bw  = abs(z(i+1) - z(i))
          hbw = 0.5_fd*bw
          if (z(i) < 1._fd+r-hbw) then
             do j = 1, ss
                res(i) = res(i) + Gimenez_s(z(i), r, nu, u, npol)
             end do
          end if
       end do
       !$omp end parallel do

    !!--- No supersampling ---
    !!
    else
       !$omp parallel do shared(nz, nu, z, r, u, npol, res) private(i) schedule(dynamic)
       do i = 1, nz
          if (z(i) < 1._fd+r) then
             res(i) = Gimenez_s(z(i), r, nu, u, npol)
          end if
       end do
       !$omp end parallel do
    end if

  contains
    pure real(8) function Gimenez_s(z, r, nu, u, npol)
      implicit none
      real(8), intent(in) :: z, r
      integer, intent(in) :: nu, npol
      real(8), intent(in), dimension(nu) :: u

      real(8), dimension(nu+1) :: a, n, Cn
      real(8) :: b, c
      integer :: i

      a  = 0._fd
      Cn = 1._fd
      n  = [(i, i=0,nu+1)]
      b  = r/(1._fd+r)
      c  = z/(1._fd+r)

      do i = 0, nu
         a(i+1) = alpha(b, c, i, npol)
      end do

      if (nu > 0) then
         Cn(1) = (1._fd - sum(u)) / (1._fd - sum(n(2:) * u / (n(2:)+2._fd)))
         Cn(2:) = u / (1._fd - n(2:) * u / (n(2:)+2._fd))
      end if
      
      Gimenez_s = 1._fd - sum(a*Cn)
    end function Gimenez_s


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
  end subroutine Gimenez
end module Gimenez_f
