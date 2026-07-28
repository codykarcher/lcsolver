      program drv_atmos
C     Emit atmos() over an altitude sweep as CSV, for verifying the port.
      implicit real (a-z)
      integer i
      write(*,'(A)') 'h,T,p,rho,a,mu'
      do i = 0, 40
        h = dble(i) * 0.5d0
        call atmos(h, T,p,rho,a,mu)
        write(*,'(F8.3,5(A,E24.16))') h,',',T,',',p,',',rho,',',a,',',mu
      enddo
      end
