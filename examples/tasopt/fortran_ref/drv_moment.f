      program drv_moment
      implicit real (a-z)
      integer icase
      write(*,'(A)') 'case,name,value'
      do icase = 1, 3
        b = 35.0d0
        bs = 12.0d0
        bo = 3.6d0
        sweep = 26.0d0
        Xaxis = 0.40d0
        lambdat = 0.25d0
        lambdas = 0.65d0
        gammat = 0.22d0
        gammas = 0.70d0
        AR = 10.1d0
        fLo = -0.3d0
        fLt = -0.05d0
        cmpo = -0.20d0
        cmps = -0.20d0
        cmpt = -0.02d0
        if (icase .eq. 2) then
          sweep = 0.0d0
          Xaxis = 0.25d0
        endif
        if (icase .eq. 3) then
          sweep = 35.0d0
          lambdat = 0.15d0
          gammat = 0.10d0
          AR = 7.5d0
        endif
        call surfcm(b,bs,bo, sweep, Xaxis,
     &              lambdat,lambdas,gammat,gammas,
     &              AR,fLo,fLt,cmpo,cmps,cmpt, CM0,CM1)
        write(*,'(I2,A,A,A,E24.16)') icase,',','CM0',',',CM0
        write(*,'(I2,A,A,A,E24.16)') icase,',','CM1',',',CM1
C       tail planform
        S = 42.0d0
        ARt = 6.0d0
        lam = 0.25d0
        qne = 12000.0d0
        CLmax = 2.0d0
        if (icase .eq. 2) then
          S = 25.0d0
          ARt = 1.8d0
          lam = 0.7d0
        endif
        if (icase .eq. 3) then
          S = 60.0d0
          ARt = 9.0d0
          lam = 0.4d0
          qne = 18000.0d0
        endif
        call tailpo(S,ARt,lam,qne,CLmax, bt,cot,pot)
        write(*,'(I2,A,A,A,E24.16)') icase,',','b',',',bt
        write(*,'(I2,A,A,A,E24.16)') icase,',','co',',',cot
        write(*,'(I2,A,A,A,E24.16)') icase,',','po',',',pot
      enddo
      end
