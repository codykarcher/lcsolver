      program drv_gasmach
      implicit real (a-z)
      integer n, i, icase
      parameter (n=5)
      real alpha(n)
      write(*,'(A)') 'case,name,value'
      do icase = 1, 5
        alpha(1) = 0.7570d0
        alpha(2) = 0.2320d0
        alpha(3) = 0.0006d0
        alpha(4) = 0.0104d0
        alpha(5) = 0.0d0
        to = 288.0d0
        po = 101325.0d0
        mo = 0.0d0
        m  = 0.8d0
        epol = 1.0d0
        if (icase .eq. 2) then
          to = 800.0d0
          m  = 0.35d0
        endif
        if (icase .eq. 3) then
          mo = 0.8d0
          m  = 0.0d0
          to = 250.0d0
        endif
        if (icase .eq. 4) then
          epol = 0.90d0
          m = 0.6d0
          to = 1400.0d0
          po = 900000.0d0
        endif
        if (icase .eq. 5) then
          mo = 0.3d0
          m  = 1.0d0
          epol = 1.05d0
        endif
        call gassum(alpha,n, to, so,so_t, ho,ho_t, cpo,ro)
        call gas_mach(alpha,n, po,to,ho,so,cpo,ro, mo,m,epol,
     &                p,t,h,s,cp,r)
        write(*,900) icase,'p  ',p
        write(*,900) icase,'t  ',t
        write(*,900) icase,'h  ',h
        write(*,900) icase,'s  ',s
        write(*,900) icase,'cp ',cp
        write(*,900) icase,'r  ',r
      enddo
 900  format(I2,',',A3,',',E24.16)
      stop
      end
