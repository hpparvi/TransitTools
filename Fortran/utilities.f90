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

real(8) function mean(n, x)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in), dimension(n) :: x
  mean = sum(x) / real(n,8)
end function mean

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


!!$def bin(x, y, weights=None, bn=None, bw=None, lim=None, minn=3, method=1): 
!!$    """
!!$    Generic routine for binning. Takes the limits of the binned space and 
!!$    either the number of bins or the width of a single bin.
!!$    
!!$    Parameters:
!!$        bn Number of bins
!!$        bw Width of a single bin
!!$        lim Limits of the binned space
!!$    """
!!$    
!!$    if bw is not None and bn is not None:
!!$        print "Error in binning: give either the number of bins or the width of a bin."
!!$    
!!$    if lim is None:
!!$        lim = np.array([x.min(), x.max()])
!!$        sw = lim[1] - lim[0] + 1e-8
!!$    else:
!!$        sw = lim[1] - lim[0]
!!$
!!$    if bw is None:
!!$        bw = sw / float(bn)
!!$    else:
!!$        bn = int(np.ceil(sw / bw))
!!$
!!$    if weights is None:
!!$        weights = np.ones(x.size, np.double) / np.double(x.size)
!!$
!!$    xb = (np.arange(bn) + 0.5) * bw + lim[0]
!!$    yb = np.zeros(bn)
!!$    ys = np.zeros(bn)
!!$    ye = np.zeros(bn)
!!$    wb = np.zeros(bn)
!!$  
!!$    bi = np.floor((x - lim[0]) / bw).astype(np.int)
!!$
!!$    # Two different binning methods for different situations. The other loops through 
!!$    # the given data, and other through the bins. The latter is probably faster when
!!$    # binning a large amount of data to relatively small number of bins.
!!$    #
!!$    # TODO: Binning - Test the efficiencies of the two binning methods.
!!$    # TODO: Binning - Weighted errorbars may still need some checking 
!!$    if method == 0:
!!$        for i, id in enumerate(bi):
!!$            yb[id] += y[i]
!!$            wb[id] += 1.
!!$        yb[wb>minn] /= wb[wb>minn]
!!$    else:   
!!$        for id in range(bn):            
!!$            m = bi == id
!!$            n = m.sum()
!!$            ws = weights[m].sum()
!!$            yb[id] = (y[m] * weights[m]).sum() / ws  # yb[id], ws = np.average(y[m], 0, weights[m], True)
!!$            ys[id] = y[m].std()
!!$            ye[id] = np.sqrt((weights[m] * (y[m] - yb[id])**2).sum() / (ws * (n-1.)))
!!$            wb[id] = n
!!$
!!$    #print ye[wb>minn] - ys[wb>minn]/np.sqrt(wb[wb>minn])
!!$
!!$    return xb[wb>minn], yb[wb>minn], ys[wb>minn], ye[wb>minn]
!!$

subroutine chi_squared(n,Xo,Xm,E,Chi)
    implicit none
    
    integer :: n
    real(8), intent(in), dimension(n) :: Xo
    real(8), intent(in), dimension(n) :: Xm
    real(8), intent(in), dimension(n) :: E

    real(8), intent(out) :: Chi

    !f2py real(8), intent(in), dimension(n) :: Xo
    !f2py real(8), intent(in), dimension(n) :: Xm
    !f2py real(8), intent(in), dimension(n) :: E
    !f2py real(8), intent(out) :: Chi
    !f2py integer, intent(in) :: n
    
    Chi = SUM(((Xo - Xm)**2 * E))
end subroutine chi_squared

subroutine planck(l,n,T,I)
    implicit none
    
    real(8), intent(in) :: l
    integer, intent(in) :: n
    real(8), intent(in), dimension(n)  :: T
    real(8), intent(out), dimension(n) :: I
        
    I = 2. * 6.6260693e-34 * 299792458.0**2 / l**5 / (exp(6.6260693e-34*299792458.0/(l*1.3806505e-23*T)) - 1.)
end subroutine planck
    
subroutine barnes_z(n,x, y, R, f, phi, z)
    implicit none

    integer :: n
    real(8), intent(in) :: R, f, phi
    real(8), intent(in), dimension(n) :: x, y
    real(8), intent(out), dimension(n) :: z
    
    !f2py integer, intent(in) :: n
    !f2py real(8), intent(in) :: R, f, phi
    !f2py real(8), intent(in), dimension(n) :: x
    !f2py real(8), intent(in), dimension(n) :: y
    !f2py real(8), intent(out), dimension(n) :: z
        
    real(8) :: sp, cp, sp2, cp2, A
    real(8), dimension(n) :: d,x2, y2, f2
        
    sp  = sin(phi)
    cp  = cos(phi)
    sp2 = sp*sp
    cp2 = cp*cp

    x2  = x*x
    y2  = y*y
    f2  = f*f
    
    A   = 1. / (1.-f)**2

    d =   4. * y2 * sp2 * cp2  * (A-1.)**2 &
     &   - 4. * (cp2 + A*sp2) * (x2 + y2 * (A*cp2 + sp2) - R*R)

    z = (-2. * y * cp * sp * (A-1.) +sqrt(d)) / (2. * (cp2 + A*sp2))
    
end subroutine barnes_z

subroutine barnes_luminosity(npt, x, y, Ms, Rs, Os, Tp, gp, f, phi, beta, c, ot, Iout)
    implicit none
    
    integer, intent(in) :: npt
    real(8), intent(in),  dimension(npt) :: x, y
    real(8), intent(in),  dimension(2)   :: c
    real(8), intent(in) :: Ms, Rs, Os, Tp, gp, beta, ot
    real(8), intent(in) :: f, phi
    
    real(8), intent(out), dimension(npt) :: Iout
    
    !f2py integer, intent(in) :: npt
    !f2py real(8), intent(in), dimension(npt) :: x
    !f2py real(8), intent(in), dimension(npt) :: y
    
    !real(8), dimension(npt,2) :: Dc
    real(8), dimension(npt,3) :: Dc, Dg, Dgg
    real(8), dimension(npt)   :: z, mu, lg2, lg, lc, gg, gc, g, T
    
    integer :: i
    
    call barnes_z(npt, x, y, Rs, f, phi, z)
    mu = z / sqrt((x**2 + y**2 + z**2))

    !! Direction vectors
    !!
    Dg(:,1) =  x
    Dg(:,2) =  y * cos(phi) + z * sin(phi)
    Dg(:,3) = -y * sin(phi) + z * cos(phi)

    Dc(:,1) = Dg(:,1)
    Dc(:,2) = 0.
    Dc(:,3) = Dg(:,3) 

    !! Lengths of the direction vectors
    !!
    lg2 = sum(Dg**2, 2)
    lg  = sqrt(lg2)
    lc  = sqrt(sum(Dc**2, 2))
    
    !! Normalize the direction vectors
    !!
    do i=1,3
        Dg(:,i) = Dg(:,i) / lg
    end do
    
    do i=1,3
        Dc(:,i) = Dc(:,i) / lc
    end do

    gg = - 6.67419e-11*Ms/lg2 
    gc =   Os*Os*lc

    do i=1,npt
        Dgg(i,:) = Dg(i,:) * gg(i) + Dc(i,:) * gc(i)
    end do
    
    g = sqrt(sum(Dgg**2,2))
    T = Tp * g**beta / gp**beta

    call planck(ot, npt, T, Iout)
    
    Iout = Iout * ( 1. - c(1) * (1. - mu) - c(2) * (1. - mu)**2)

end subroutine barnes_luminosity
