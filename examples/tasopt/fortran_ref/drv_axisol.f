      program drv_axisol
c---- Potential flow over a 737-like fuselage, point-tail and edge-tail,
c---- at two Mach numbers.
      implicit real (a-h,m,o-z)
      integer nldim, nl, ilte, nc, iclose, icase, i
      parameter (nldim = 200)
      real xl(nldim), zl(nldim), sl(nldim), dyl(nldim), ql(nldim)
      write(*,'(A)') 'case,i,x,z,s,dy,q'
      do 500 icase = 1, 4
        xnose = 0.0
        xend  = 37.8
        xblend1 = 6.1
        xblend2 = 30.0
        Amax = 12.0
        anose = 1.8
        btail = 1.6
        iclose = 0
        rMach = 0.80
        nc = 30
        if (icase .eq. 2) iclose = 1
        if (icase .eq. 3) rMach = 0.20
        if (icase .eq. 4) then
          nc = 20
          anose = 2.2
          btail = 2.0
        endif
        call axisol(xnose,xend,xblend1,xblend2, Amax,
     &              anose,btail,iclose, rMach, nc,
     &              nldim, nl, ilte, xl,zl,sl,dyl,ql)
        write(*,901) icase, 0, float(nl), float(ilte), 0.0, 0.0, 0.0
        do i = 1, nl
          write(*,901) icase, i, xl(i), zl(i), sl(i), dyl(i), ql(i)
        enddo
 500  continue
 901  format(I2,',',I4,',',4(E24.16,','),E24.16)
      stop
      end
