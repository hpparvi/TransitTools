real(8) function mean(n, x)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in), dimension(n) :: x
  mean = sum(x) / real(n,8)
end function mean


subroutine running_mean(nx,x,width,res)
  implicit none
  integer :: nx, width
  
  real(8), intent(in),  dimension(nx) :: x
  real(8), intent(out), dimension(nx) :: res

  integer :: np, hw, i
  real(8) :: iw

  iw  = 1./real(width,8)
  hw  = (width-1)/2
  np  = nx - width + 1
  res = 0.

  do i = 1+hw, nx-hw
     res(i) = sum(x(i-hw:i+hw)) * iw
  end do
end subroutine running_mean


!!$subroutine fold(n, x, period, origo, shift, normalize, clip_range, x_out)
!!$  implicit none
!!$  
!!$  integer :: n
!!$  real(8), dimension(n), intent(in) :: x
!!$  real(8), intent(in) :: period, origo
!!$  real(8), dimension(n), intent(out) :: x_out
!!$
!!$end subroutine fold


subroutine bin(n, x, y, nbins, xb, yb, ye)
  use omp_lib
  implicit none

  integer :: n, nbins
  real(8), dimension(n), intent(in) :: x, y
  real(8), dimension(nbins), intent(out) :: xb, yb, ye
  integer, dimension(n) :: bid, bweight
  logical, dimension(n) :: mask

  real(8) :: xw, bw, xl, xh
  integer :: i, nb

  xl = minval(x)
  xh = maxval(x)
  xw = xh - xl + 1e-8
  bw = xw / real(nbins,8)

  xb  = ([(i, i=0,nbins-1)] + 0.5) * bw + xl
  yb  = 0.
  bweight  = 0.
  bid = floor((x - xl) / bw) + 1

  !$omp parallel do shared(nbins, bid, bweight, yb, ye) private(i, mask, nb)
  do i=1,nbins
     mask = bid == i
     nb = count(mask)
     bweight(i) = nb
     yb(i) = sum(y, mask=mask) / real(nb,8)
     ye(i) = sqrt(sum((pack(y,mask) - yb(i))**2) / real(nb*(nb-1),8))
  end do
  !$omp end parallel do
  end subroutine bin


subroutine bin_noerr(n, x, y, nbins, xb, yb)
  implicit none

  integer :: n, nbins
  real(8), dimension(n), intent(in) :: x, y
  real(8), dimension(nbins), intent(out) :: xb, yb
  integer, dimension(n) :: bid, bweight
  logical, dimension(n) :: mask

  real(8) :: xw, bw, xl, xh
  integer :: i, nb

  xl = minval(x)
  xh = maxval(x)
  xw = xh - xl + 1e-8
  bw = xw / real(nbins,8)

  xb  = ([(i, i=0,nbins-1)] + 0.5) * bw + xl
  yb  = 0.
  bweight  = 0.
  bid = floor((x - xl) / bw) + 1

  do i=1,nbins
     mask = bid == i
     nb = count(mask)
     bweight(i) = nb
     yb(i) = sum(y, mask=mask) / real(nb,8)
  end do
end subroutine bin_noerr


subroutine chi_squared(n,Xo,Xm,E,Chi)
    implicit none
    
    integer :: n
    real(8), intent(in), dimension(n) :: Xo
    real(8), intent(in), dimension(n) :: Xm
    real(8), intent(in), dimension(n) :: E

    real(8), intent(out) :: Chi
   
    Chi = sum(((Xo - Xm)**2 * E))
end subroutine chi_squared
