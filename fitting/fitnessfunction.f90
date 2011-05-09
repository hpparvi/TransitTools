module fitnessfunction
  use omp_lib
  use, intrinsic :: ISO_C_BINDING
  implicit none

  include "mpif.h"

  integer, parameter :: FS = C_FLOAT
  integer, parameter :: FD = C_DOUBLE

contains
  subroutine calculate_time_with_ttv(amplitude, ttv_period, period, time, transit_number, time_out, n)
    implicit none
    real(8), dimension(n), intent(in) :: transit_number, time
    real(8), intent(in) :: amplitude, ttv_period, period
    real(8), dimension(n), intent(out) :: time_out
    integer, intent(in) :: n
    integer :: i
    
    !$omp parallel do private(i) shared(amplitude, ttv_period, period, time, transit_number, time_out, n) &
    !$omp schedule(static)
    do i=1,n
       time_out(i) = time(i) + amplitude*sin(6.283185307179586_fd * ttv_period * period * transit_number(i))
    end do
    !$omp end parallel do
  end subroutine calculate_time_with_ttv
  
  real(8) function chi_sqr(observed, model, var_inv, n)
    implicit none
    real(8), dimension(n), intent(in) :: observed, model, var_inv
    integer, intent(in) :: n
    integer :: i

    chi_sqr = 0._fd
    !$omp parallel do private(i) shared(observed, model, var_inv, n) &
    !$omp schedule(static) &
    !$omp reduction(+:chi_sqr)
    do i=1,n
       chi_sqr = chi_sqr + (observed(i) - model(i))**2 * var_inv(i)
    end do
  end function chi_sqr

  subroutine apply_zeropoints(zeropoints, chn_l, model, nchn, npt, res)
    implicit none
    integer, intent(in) :: nchn, npt
    real(8), intent(in), dimension(npt)  :: model
    real(8), intent(in), dimension(nchn) :: zeropoints
    integer, intent(in), dimension(nchn) :: chn_l
    real(8), intent(out), dimension(npt) :: res
    integer j, i, i_s, i_e

    i_s = 0
    res(1:chn_l(1)) = zeropoints(1) * model(1:chn_l(1))
    do j=2,nchn
       i_s = i_s + chn_l(j-1)
       i_e = i_s + chn_l(j)
      res(i_s:i_e) = zeropoints(j) * model(i_s:i_e)
    end do
  end subroutine apply_zeropoints

end module fitnessfunction
