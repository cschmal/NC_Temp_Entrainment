module mod_forced_hong_model
    implicit none
    real(8), save :: c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22
    contains
        subroutine ode(y, t, yprime)
            real(8), dimension(9), intent(in) :: y
            real(8), intent(in) :: t
            real(8), dimension(9), intent(out) :: yprime
            real(8) :: pi,coeff,arg,force
            pi=acos(-1.)
            coeff=c3*c0**2/(2*pi*(c1*c0))
            arg=y(1)*cos(2.*pi*c4/c0)+y(2)*sin(2.*pi*c4/c0)-cos(pi*c1)
            force=c2*(atan(coeff*arg)/pi)
            yprime(1)=y(1)*(1-y(1)**2-y(2)**2)-2*pi*y(2)/c0
            yprime(2)=y(2)*(1-y(1)**2-y(2)**2)+2*pi*y(1)/c0
            yprime(3)=((1+force)*c5*(y(8)**c22)/(c20+(y(8)**c22)))-(c8*y(3)) 
            yprime(4)=c6*y(3)-((c7+c9)*y(4))
            yprime(5)=(c7*y(4))+(c18*y(9))-(y(5)*(c10+(c17*y(8))))
            yprime(6)=c11-(c14*y(6))
            yprime(7)=(c12*y(4)*y(6)/(c21+y(4)))-((c13+c15)*y(7))
            yprime(8)=(c13*y(7))-(y(8)*(c16+(c17*y(5))))+(c18*y(9))
            yprime(9)=c17*y(5)*y(8)-((c18+c19)*y(9))
       end subroutine ode
end module mod_forced_hong_model
