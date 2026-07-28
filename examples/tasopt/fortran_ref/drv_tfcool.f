      program drv_tfcool
c---- mcool and its inverse Tmcalc, across cooled-row counts and temperatures.
      implicit real (a-z)
      integer icase, ncrowx, ncrow, i
      parameter (ncrowx = 4)
      real Tmrow(ncrowx), epsrow(ncrowx)
      real epsrow_Tt3(ncrowx), epsrow_Tt4(ncrowx), epsrow_Trr(ncrowx)
      real Tmback(ncrowx)
      write(*,'(A)') 'case,name,value'
      do icase = 1, 5
        do i = 1, ncrowx
          Tmrow(i) = 1200.0d0
        enddo
        Tt3 = 700.0d0
        Tt4 = 1600.0d0
        dTstreak = 200.0d0
        Trrat = 0.90d0
        efilm = 0.70d0
        tfilm = 0.30d0
        StA = 0.09d0
        if (icase .eq. 2) Tt4 = 1900.0d0
        if (icase .eq. 3) then
c-------- cool enough that fewer rows need cooling
          Tt4 = 1250.0d0
          dTstreak = 50.0d0
        endif
        if (icase .eq. 4) then
c-------- staggered metal temperatures down the turbine
          Tmrow(1) = 1250.0d0
          Tmrow(2) = 1200.0d0
          Tmrow(3) = 1150.0d0
          Tmrow(4) = 1100.0d0
          Trrat = 0.85d0
        endif
        if (icase .eq. 5) then
          efilm = 0.50d0
          tfilm = 0.45d0
          StA = 0.15d0
          Tt3 = 600.0d0
        endif
        call mcool(ncrowx,ncrow,
     &             Tmrow,Tt3,Tt4,dTstreak, Trrat,
     &             efilm,tfilm,StA,
     &             epsrow, epsrow_Tt3, epsrow_Tt4, epsrow_Trr)
        write(*,'(I2,A,A,A,E24.16)') icase,',','ncrow',',',float(ncrow)
        do i = 1, ncrowx
          write(*,900) icase,'eps',i,epsrow(i)
          write(*,900) icase,'dT3',i,epsrow_Tt3(i)
          write(*,900) icase,'dT4',i,epsrow_Tt4(i)
          write(*,900) icase,'dTr',i,epsrow_Trr(i)
        enddo
c------ round trip: recover the metal temperatures from the coolant flows
        do i = 1, ncrowx
          Tmback(i) = 0.0d0
        enddo
        call Tmcalc(ncrowx,ncrow,
     &              Tmback,Tt3,Tt4,dTstreak, Trrat,
     &              efilm,tfilm,StA, epsrow)
        do i = 1, ncrowx
          write(*,900) icase,'Tm ',i,Tmback(i)
        enddo
      enddo
 900  format(I2,',',A3,I1,',',E24.16)
      stop
      end
