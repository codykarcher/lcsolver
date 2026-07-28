      program drv_fusew
C     Drive fusew() over a few configurations. All actual arguments are
C     named variables -- see README on literal-constant passing.
      implicit real (a-z)
      integer icase
      write(*,'(A)') 'case,tskin,tcone,tfweb,tfloor,xhbend,xvbend,'//
     &  'EIhshell,EIhbend,EIvshell,EIvbend,GJshell,GJcone,Wshell,'//
     &  'Wcone,Wwindow,Winsul,Wfloor,Whbend,Wvbend,Wfuse,xWfuse,cabVol'
      do icase = 1, 3
        gee = 9.81d0
        Nland = 6.0d0
        Wfix = 13000.0d0
        Wpay = 175000.0d0
        Wpadd = 30000.0d0
        Wseat = 25000.0d0
        Wapu = 5000.0d0
        Weng = 0.0d0
        fstring = 0.35d0
        fframe = 0.25d0
        ffadd = 0.20d0
        deltap = 55000.0d0
        Wpwindow = 435.0d0
        Wppinsul = 22.0d0
        Wppfloor = 60.0d0
        Whtail = 12000.0d0
        Wvtail = 9000.0d0
        rMh = 0.4d0
        rMv = 0.7d0
        Lhmax = 200000.0d0
        Lvmax = 180000.0d0
        bv = 8.0d0
        lambdav = 0.3d0
        nvtail = 1.0d0
        Rfuse = 1.9d0
        dRfuse = 0.3d0
        wfb = 0.0d0
        nfweb = 0.0d0
        lambdac = 0.3d0
        xnose = 0.0d0
        xshell1 = 5.0d0
        xshell2 = 31.0d0
        xconend = 36.0d0
        xhtail = 34.0d0
        xvtail = 33.0d0
        xwing = 18.0d0
        xwbox = 18.0d0
        cbox = 3.0d0
        xfix = 3.0d0
        xapu = 35.0d0
        xeng = 16.0d0
        hfloor = 0.13d0
        sigskin = 1.5d8
        sigbend = 1.5d8
        rhoskin = 2700.0d0
        rhobend = 2700.0d0
        Eskin = 6.9d10
        Ebend = 6.9d10
        Gskin = 2.4d10
C       case 2: multi-bubble; case 3: bigger tube, higher loads
        if (icase .eq. 2) then
          wfb = 0.4d0
          nfweb = 1.0d0
          Rfuse = 1.7d0
        endif
        if (icase .eq. 3) then
          Rfuse = 2.6d0
          dRfuse = 0.0d0
          Wpay = 350000.0d0
          deltap = 60000.0d0
          xshell2 = 45.0d0
          xconend = 52.0d0
          xhtail = 49.0d0
          xvtail = 48.0d0
          xwing = 26.0d0
          xwbox = 26.0d0
        endif
        call fusew(gee,Nland,Wfix,Wpay,Wpadd,Wseat,Wapu,Weng,
     &             fstring,fframe,ffadd,deltap,
     &             Wpwindow,Wppinsul,Wppfloor,
     &             Whtail,Wvtail,rMh,rMv,Lhmax,Lvmax,
     &             bv,lambdav,nvtail,
     &             Rfuse,dRfuse,wfb,nfweb,lambdac,
     &             xnose,xshell1,xshell2,xconend,
     &             xhtail,xvtail,
     &             xwing,xwbox,cbox,
     &             xfix,xapu,xeng,
     &             hfloor,
     &             sigskin,sigbend, rhoskin,rhobend,
     &             Eskin,Ebend,Gskin,
     &             tskin, tcone, tfweb, tfloor, xhbend, xvbend,
     &             EIhshell,EIhbend, EIvshell,EIvbend,
     &             GJshell ,GJcone,
     &             Wshell, Wcone, Wwindow, Winsul, Wfloor,
     &             Whbend, Wvbend, Wfuse, xWfuse, cabVol)
        write(*,'(I2,22(A,E24.16))') icase,
     &    ',',tskin,',',tcone,',',tfweb,',',tfloor,',',xhbend,
     &    ',',xvbend,',',EIhshell,',',EIhbend,',',EIvshell,
     &    ',',EIvbend,',',GJshell,',',GJcone,',',Wshell,',',Wcone,
     &    ',',Wwindow,',',Winsul,',',Wfloor,',',Whbend,',',Wvbend,
     &    ',',Wfuse,',',xWfuse,',',cabVol
      enddo
      end
