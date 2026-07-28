      program drv_tfsize
c---- On-design sizing of a CFM56-class turbofan at top of climb.
      implicit real (a-z)
      integer iBLIc, ifuel, icool, ncrowx, ncrow, icase, i
      parameter (ncrowx = 4)
      real epsrow(ncrowx), Tmrow(ncrowx)
      logical Lconv
      write(*,'(A)') 'case,name,value'
      do icase = 1, 3
        gee = 9.81d0
        M0 = 0.80d0
        T0 = 219.43d0
        p0 = 23842.0d0
        a0 = sqrt(1.4d0*287.0d0*T0)
        M2 = 0.60d0
        M25 = 0.60d0
        Feng = 25000.0d0
        Phiinl = 0.0d0
        Kinl = 0.0d0
        iBLIc = 0
        BPR = 5.1d0
        pif = 1.685d0
        pilc = 1.935d0
        pihc = 9.369d0
        pid = 0.998d0
        pib = 0.94d0
        pifn = 0.98d0
        pitn = 0.989d0
        Tt4 = 1450.0d0
        Ttf = 280.0d0
        ifuel = 24
        etab = 0.985d0
        epf0 = 0.8948d0
        eplc0 = 0.88d0
        ephc0 = 0.87d0
        epht0 = 0.889d0
        eplt0 = 0.899d0
        pifK = 1.685d0
        epfK = 0.0d0
        mofft = 0.0d0
        Pofft = 0.0d0
        Tt9 = 300.0d0
        pt9 = 30000.0d0
        epsl = 0.0d0
        epsh = 0.0d0
        icool = 0
        Mtexit = 1.0d0
        dTstrk = 200.0d0
        StA = 0.09d0
        efilm = 0.7d0
        tfilm = 0.3d0
        M4a = 0.9d0
        ruc = 0.9d0
        do i = 1, ncrowx
          epsrow(i) = 0.0d0
          Tmrow(i) = 1200.0d0
        enddo
        if (icase .eq. 2) then
c-------- with cooling: Tmrow given, epsrow computed
          icool = 2
        endif
        if (icase .eq. 3) then
c-------- cooling plus mass and power offtakes (multi-pass loop)
          icool = 2
          mofft = 0.5d0
          Pofft = 60000.0d0
          Tt4 = 1550.0d0
        endif
        call tfsize(gee,M0,T0,p0,a0, M2,M25,
     &    Feng, Phiinl,Kinl,iBLIc,
     &    BPR,pif,pilc,pihc, pid,pib,pifn,pitn,
     &    Ttf,ifuel,etab,
     &    epf0,eplc0,ephc0,epht0,eplt0, pifK,epfK,
     &    mofft,Pofft, Tt9,pt9, epsl,epsh, icool,
     &    Mtexit,dTstrk,StA,efilm,tfilm, M4a,ruc,
     &    ncrowx, ncrow, epsrow,Tmrow,
     &    TSFC,Fsp,hfuel,ff,mcore,
     &    Tt0 ,ht0 ,pt0 ,cpt0 ,Rt0 ,
     &    Tt18,ht18,pt18,cpt18,Rt18,
     &    Tt19,ht19,pt19,cpt19,Rt19,
     &    Tt2 ,ht2 ,pt2 ,cpt2 ,Rt2 ,
     &    Tt21,ht21,pt21,cpt21,Rt21,
     &    Tt25,ht25,pt25,cpt25,Rt25,
     &    Tt3 ,ht3 ,pt3 ,cpt3 ,Rt3 ,
     &    Tt4, ht4 ,pt4 ,cpt4 ,Rt4 ,
     &    Tt41,ht41,pt41,cpt41,Rt41,
     &    Tt45,ht45,pt45,cpt45,Rt45,
     &    Tt49,ht49,pt49,cpt49,Rt49,
     &    Tt5 ,ht5 ,pt5 ,cpt5 ,Rt5 ,
     &    Tt7 ,ht7 ,pt7 ,cpt7 ,Rt7 ,
     &    u0 ,
     &    T2 ,u2 ,p2 ,cp2 ,R2 ,A2 ,
     &    T25,u25,p25,cp25,R25,A25,
     &    T5 ,u5 ,p5 ,cp5 ,R5 ,A5 ,
     &    T6 ,u6 ,p6 ,cp6 ,R6 ,A6 ,
     &    T7 ,u7 ,p7 ,cp7 ,R7 ,A7 ,
     &    T8 ,u8 ,p8 ,cp8 ,R8 ,A8 ,
     &    u9 , A9 ,
     &    epf ,eplc ,ephc ,epht ,eplt ,
     &    etaf,etalc,etahc,etaht,etalt, Lconv)
        write(*,900) icase,'TSFC ',TSFC
        write(*,900) icase,'Fsp  ',Fsp
        write(*,900) icase,'ff   ',ff
        write(*,900) icase,'mcore',mcore
        write(*,900) icase,'Tt3  ',Tt3
        write(*,900) icase,'Tt41 ',Tt41
        write(*,900) icase,'Tt45 ',Tt45
        write(*,900) icase,'Tt49 ',Tt49
        write(*,900) icase,'pt3  ',pt3
        write(*,900) icase,'pt45 ',pt45
        write(*,900) icase,'A2   ',A2
        write(*,900) icase,'A25  ',A25
        write(*,900) icase,'A5   ',A5
        write(*,900) icase,'A7   ',A7
        write(*,900) icase,'u6   ',u6
        write(*,900) icase,'u8   ',u8
        write(*,900) icase,'etaf ',etaf
        write(*,900) icase,'etaht',etaht
      enddo
 900  format(I2,',',A5,',',E24.16)
      stop
      end
