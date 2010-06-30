from string import Template
def construct_Gimenez(npoints, npol, nldcoeff, nthreads=0, build_module=False):
    
    def construct_main(npoints, nldcoeff, nthreads):
        code = Template("""
        subroutine Gimenez(z, r, u, result)
            use omp_lib
            implicit none
            real(8), intent(in), dimension(${npoints}) :: z
            real(8), intent(in), dimension(${nldc}) :: u
            real(8), intent(in) :: r
            real(8), intent(out), dimension(${npoints}) :: result
            integer :: i
            
            result = 1._fd
            
            !$$ if (${nthreads} /= 0) call omp_set_num_threads(${nthreads})
            !$$omp parallel do shared(z, r, u, result) private(i) schedule(dynamic)
            do i = 1, ${npoints}
               if (z(i) < 1._fd+r) then
                  result(i) = Gimenez_s(z(i), r, u)
               end if
            end do
            !$$omp end parallel do
        end subroutine Gimenez
        """)
        
        return code.substitute(npoints=npoints, nthreads=nthreads, nldc=nldcoeff)
    
    def construct_Gimenez_s(nldcoeff):
        code = """
        pure real(8) function Gimenez_s(z, r, u)
            implicit none
            real(8), intent(in) :: z, r
            real(8), intent(in), dimension(${nldc}) :: u
            real(8), dimension(${nldc}+1) :: a, n, Cn
            real(8) :: b, c, f
            integer :: i
            
            a  = 0._fd
            Cn = 1._fd
            n = [(i, i=0,${nldc})]
            
            b = r/(1._fd+r)
            c = z/(1._fd+r)
            
            do i = 0, ${nldc}
                a(i+1) = alpha(b, c, i)
            end do
        """
            
        if nldcoeff > 0:
            code += """
            Cn(1) = (1._fd - sum(u)) / (1._fd - sum(n(2:) * u / (n(2:)+2._fd)))
            Cn(2:) = u / (1._fd - n(2:) * u / (n(2:)+2._fd))
            """
            
        code += """
            Gimenez_s = 1._fd - sum(a*Cn)
        end function Gimenez_s
        """
    
        return Template(code).substitute(nldc=nldcoeff)
    
    def construct_alpha(n):
        
        code = Template("""
        pure real(8) function alpha(b, c, n)
            implicit none
            real(8), intent(in) :: b, c
            integer, intent(in) :: n
            
            real(8)  :: norm, sum, vl
            real(8), dimension(${npol}) :: e, d
            
            real(8) :: nu, nm
            integer :: i
            
            nu = (n+2._fd)/2._fd
            norm = b*b * (1._fd - c*c)**(1._fd + nu) / (nu * GAMMA(1._fd + nu))
            
            call jacobi_Burkardt(0._fd, 1._fd+nu, 1._fd-2._fd*c*c, d)
            call jacobi_Burkardt(   nu,    1._fd, 1._fd-2._fd*(1._fd-b), e)
            
            sum = 0._fd
            do i = 0, ${npol}-1
                nm = LOG_GAMMA(nu+i+1.) - LOG_GAMMA(i+2._fd)
                vl = (-1)**i * (2._fd+2._fd*i+nu)*exp(nm)
                nm = LOG_GAMMA(i+1._fd) + LOG_GAMMA(nu+1._fd) - LOG_GAMMA(i+1._fd+nu)
                
                e(i+1) = e(i+1) * exp(nm)
                vl   = vl * d(i+1) * e(i+1) * e(i+1)
                sum  = sum + vl
            end do
            alpha = norm * sum
        end function alpha
        """)
    
        return code.substitute(npol=npol)
    
    def construct_jacobi(npol):
        code = """
            pure subroutine jacobi_Burkardt(alpha, beta, x, cx)
            implicit none
            real(8), intent(in) :: alpha, beta, x
            real(8), intent(out), dimension(:) :: cx
            real(8) :: ri, c1, c2, c3, c4
            integer :: i
            
            cx(1) = 1._fd
        """
        
        if npol > 0:
            code += Template("""
             cx(2) = ( 1._fd + 0.5_fd * ( alpha + beta ) ) * x + 0.5_fd * ( alpha - beta )
            
             do i = 2, ${npolm}
                ri = real(i,8)
                c1 = 2._fd * ri * ( ri + alpha + beta ) * ( 2._fd * ri - 2._fd + alpha + beta )
                c2 = ( 2._fd* ri - 1._fd + alpha + beta ) * ( 2._fd * ri  + alpha + beta ) * ( 2._fd* ri - 2._fd + alpha + beta )
                c3 = ( 2._fd* ri - 1._fd + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
                c4 = - 2._fd * ( ri - 1._fd + alpha ) * ( ri - 1._fd + beta )  * ( 2._fd* ri + alpha + beta )
            
                cx(i+1) = ( ( c3 + c2 * x ) * cx(i) + c4 * cx(i-1) ) / c1
             end do
            """).substitute(npolm=npol-1)
            
        code += """
            end subroutine jacobi_Burkardt
            """
            
        return code
    
    code = ""
    
    if build_module:
        code += """
        module Gimenez_model
            use, intrinsic :: ISO_C_BINDING
            implicit none
            integer, parameter :: FS = C_FLOAT
            integer, parameter :: FD = C_DOUBLE
            
            contains
        """
    
    code += construct_main(npoints, nldcoeff, nthreads)
    code += construct_Gimenez_s(nldcoeff)
    code += construct_jacobi(npol)
    code += construct_alpha(npol)
    
    if build_module:
        code += "end module Gimenez_model"
    
    return code

f = open('tmp.f90', 'w')
f.write(construct_Gimenez(100, 10, 2, 4, True))
f.close()
