      program drv_wingpo
      implicit real (a-z)
      integer icase
      write(*,'(A)') 'case,name,value'
      do icase = 1, 5
        b = 35.0d0
        bs = 12.0d0
        bo = 3.6d0
        lambdat = 0.25d0
        lambdas = 0.65d0
        gammat = 0.22d0
        gammas = 0.70d0
        AR = 11.667d0
        N = 3.0d0
        W = 700000.0d0
        Lhtail = -20000.0d0
        fLo = -0.3d0
        fLt = -0.05d0
        sweep = 26.0d0
        CL = 0.55d0
        CLhtail = -0.05d0
        duo = 0.0d0
        dus = 0.0d0
        dut = 0.0d0
        if (icase .eq. 2) then
          sweep = 0.0d0
          Lhtail = 0.0d0
        endif
        if (icase .eq. 3) then
          lambdat = 0.15d0
          gammat = 0.10d0
          AR = 7.5d0
          N = 2.5d0
        endif
        if (icase .eq. 4) then
          duo = 0.018d0
          dus = 0.014d0
          dut = 0.009d0
          sweep = 35.0d0
        endif
        if (icase .eq. 5) then
          fLo = 0.0d0
          fLt = 0.0d0
          CL = 0.80d0
          W = 1200000.0d0
        endif
        call wingpo(b,bs,bo,lambdat,lambdas,gammat,gammas,
     &              AR,N,W,Lhtail,fLo,fLt, po)
        call wingcl(b,bs,bo,lambdat,lambdas,gammat,gammas,
     &              sweep,AR,CL,CLhtail,fLo,fLt,
     &              duo,dus,dut, clo,cls,clt)
        write(*,900) icase,'po ',po
        write(*,900) icase,'clo',clo
        write(*,900) icase,'cls',cls
        write(*,900) icase,'clt',clt
      enddo
 900  format(I2,',',A3,',',E24.16)
      stop
      end
