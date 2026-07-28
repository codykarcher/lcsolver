      program drv_gasburn
c---- gasburn.f's `gasprop` across three fuels and two burner temperatures.
      implicit real (a-z)
      integer icase, nh, nc, nn, no
      real alphai(5)
      write(*,'(A)') 'case,name,value'
      do icase = 1, 6
c------ dry air by mass: 75.7% N2, 23.2% O2, small CO2/H2O
        alphai(1) = 0.7570d0
        alphai(2) = 0.2320d0
        alphai(3) = 0.0006d0
        alphai(4) = 0.0104d0
        alphai(5) = 0.0d0
        ttf = 300.0d0
        tt3 = 700.0d0
        tt4 = 1400.0d0
        rfuel = 519.65d0
        cpfuel = 2240.0d0
        hsfuel = -4.675d6
c------ heavy hydrocarbon, the TASOPT default
        nh = 2
        nc = 1
        nn = 0
        no = 0
        if (icase .eq. 2) tt4 = 1700.0d0
        if (icase .eq. 3) then
c-------- methane
          nh = 4
          nc = 1
        endif
        if (icase .eq. 4) then
c-------- propane
          nh = 8
          nc = 3
        endif
        if (icase .eq. 5) then
c-------- cold compressor exit, hot burner
          tt3 = 500.0d0
          tt4 = 1800.0d0
          ttf = 435.0d0
        endif
        if (icase .eq. 6) then
c-------- fuel carrying oxygen and nitrogen
          nh = 6
          nc = 2
          nn = 1
          no = 1
          cpfuel = 1900.0d0
          hsfuel = -6.20d6
        endif
        call gasprop(ttf,tt3,tt4,
     &               rfuel,cpfuel,hsfuel,
     &               nh,nc,nn,no,
     &               alphai,
     &               cp3,gam3, cp4,gam4, f)
        write(*,'(I2,A,A,A,E24.16)') icase,',','f',',',f
        write(*,'(I2,A,A,A,E24.16)') icase,',','cp3',',',cp3
        write(*,'(I2,A,A,A,E24.16)') icase,',','gam3',',',gam3
        write(*,'(I2,A,A,A,E24.16)') icase,',','cp4',',',cp4
        write(*,'(I2,A,A,A,E24.16)') icase,',','gam4',',',gam4
        write(*,'(I2,A,A,A,E24.16)') icase,',','a1',',',alphai(1)
        write(*,'(I2,A,A,A,E24.16)') icase,',','a2',',',alphai(2)
        write(*,'(I2,A,A,A,E24.16)') icase,',','a3',',',alphai(3)
        write(*,'(I2,A,A,A,E24.16)') icase,',','a4',',',alphai(4)
        write(*,'(I2,A,A,A,E24.16)') icase,',','a5',',',alphai(5)
      enddo
      stop
      end
