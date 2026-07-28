      program drv_ecmap
      implicit real (a-z)
      integer icase, i
      real Cmap(9)
      write(*,'(A)') 'case,name,value'
      do icase = 1, 6
c------ historical fan set, which has nonzero CK/DK
        Cmap(1)=3.50d0
        Cmap(2)=0.80d0
        Cmap(3)=0.03d0
        Cmap(4)=0.75d0
        Cmap(5)=-0.50d0
        Cmap(6)=3.0d0
        Cmap(7)=6.0d0
        Cmap(8)=2.5d0
        Cmap(9)=15.0d0
        piD = 1.60d0
        mbD = 200.0d0
        effo = 0.90d0
        piK = 1.60d0
        effK = 0.0d0
        pi = 1.60d0
        mb = 200.0d0
        if (icase .eq. 2) mb = 170.0d0
        if (icase .eq. 3) mb = 230.0d0
        if (icase .eq. 4) pi = 1.45d0
        if (icase .eq. 5) then
          pi = 1.75d0
          mb = 185.0d0
          effK = -0.02d0
        endif
        if (icase .eq. 6) then
c-------- the ACTIVE shipped set: CK = DK = 0
          Cmap(4)=0.95d0
          Cmap(8)=0.0d0
          Cmap(9)=0.0d0
          pi = 1.72d0
          mb = 210.0d0
          effK = -0.015d0
        endif
        call ecmap(pi,mb, piD,mbD, Cmap, effo, piK, effK,
     &             eff, eff_pi, eff_mb)
        write(*,900) icase,'eff',eff
        write(*,900) icase,'dpi',eff_pi
        write(*,900) icase,'dmb',eff_mb
      enddo
 900  format(I2,',',A3,',',E24.16)
      stop
      end
