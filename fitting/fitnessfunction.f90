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
       time_out(i) = amplitude*sin(6.283185307179586_fd * ttv_period * period * transit_number(i))
       time_out(i) = time(i) + time_out(i)
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

end module fitnessfunction
