module barnes_f
  use, intrinsic :: ISO_C_BINDING

  implicit none

  integer, parameter :: FS = C_FLOAT
  integer, parameter :: FD = C_DOUBLE

contains

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

end module barnes_f
