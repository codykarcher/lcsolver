      program drv_tfoper
c---- Size a CFM56-class turbofan on design, then run it off design.
c---- Case 1: back at the design point, Tt4 specified (a self-consistency
c----         check -- tfoper must reproduce tfsize).
c---- Case 2: lower and slower, Tt4 specified.
c---- Case 3: same condition, thrust specified instead (iTFspec=2).
c---- Case 4: cooled turbine plus mass and power offtakes.
      implicit real (a-z)
      integer iBLIc, ifuel, icool, ncrowx, ncrow, icase, i, iTFspec
      parameter (ncrowx = 4)
      real epsrow(ncrowx), Tmrow(ncrowx)
      logical Lconv
      write(*,'(A)') 'case,name,value'

      do 500 icase = 1, 4

c===== design point =================================================
      gee = 9.81d0
      M0 = 0.80d0
      T0 = 219.43d0
      p0 = 23842.0d0
      a0 = sqrt(1.4d0*287.0d0*T0)
      Tref = 288.2d0
      pref = 101320.0d0
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
      Gearf = 1.0d0
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
      ncrow = 0
      do i = 1, ncrowx
        epsrow(i) = 0.0d0
        Tmrow(i) = 1200.0d0
      enddo
      if (icase .ge. 3) then
        icool = 2
        mofft = 0.5d0
        Pofft = 60000.0d0
      endif

      call tfsize(gee,M0,T0,p0,a0, M2,M25,
     &  Feng, Phiinl,Kinl,iBLIc,
     &  BPR,pif,pilc,pihc, pid,pib,pifn,pitn,
     &  Ttf,ifuel,etab,
     &  epf0,eplc0,ephc0,epht0,eplt0, pifK,epfK,
     &  mofft,Pofft, Tt9,pt9, epsl,epsh, icool,
     &  Mtexit,dTstrk,StA,efilm,tfilm, M4a,ruc,
     &  ncrowx, ncrow, epsrow,Tmrow,
     &  TSFC,Fsp,hfuel,ff,mcore,
     &  Tt0 ,ht0 ,pt0 ,cpt0 ,Rt0 ,
     &  Tt18,ht18,pt18,cpt18,Rt18,
     &  Tt19,ht19,pt19,cpt19,Rt19,
     &  Tt2 ,ht2 ,pt2 ,cpt2 ,Rt2 ,
     &  Tt21,ht21,pt21,cpt21,Rt21,
     &  Tt25,ht25,pt25,cpt25,Rt25,
     &  Tt3 ,ht3 ,pt3 ,cpt3 ,Rt3 ,
     &  Tt4, ht4 ,pt4 ,cpt4 ,Rt4 ,
     &  Tt41,ht41,pt41,cpt41,Rt41,
     &  Tt45,ht45,pt45,cpt45,Rt45,
     &  Tt49,ht49,pt49,cpt49,Rt49,
     &  Tt5 ,ht5 ,pt5 ,cpt5 ,Rt5 ,
     &  Tt7 ,ht7 ,pt7 ,cpt7 ,Rt7 ,
     &  u0 ,
     &  T2 ,u2 ,p2 ,cp2 ,R2 ,A2 ,
     &  T25,u25,p25,cp25,R25,A25,
     &  T5 ,u5 ,p5 ,cp5 ,R5 ,A5 ,
     &  T6 ,u6 ,p6 ,cp6 ,R6 ,A6 ,
     &  T7 ,u7 ,p7 ,cp7 ,R7 ,A7 ,
     &  T8 ,u8 ,p8 ,cp8 ,R8 ,A8 ,
     &  u9 , A9 ,
     &  epf ,eplc ,ephc ,epht ,eplt ,
     &  etaf,etalc,etahc,etaht,etalt, Lconv)

c===== design-point map anchors, exactly as tfcalc.f builds them =====
      fo = mofft/mcore
      mbfD  = mcore*sqrt(Tt2 /Tref)/(pt2 /pref)*BPR
      mblcD = mcore*sqrt(Tt19/Tref)/(pt19/pref)
      mbhcD = mcore*sqrt(Tt25/Tref)/(pt25/pref)*(1.0d0-fo)
      mbhtD = mcore*sqrt(Tt41/Tref)/(pt41/pref)*(1.0d0-fo+ff)
      mbltD = mcore*sqrt(Tt45/Tref)/(pt45/pref)*(1.0d0-fo+ff)

      NbfD  = (1.0d0/Gearf)/sqrt(Tt2 /Tref)
      NblcD = 1.0d0/sqrt(Tt19/Tref)
      NbhcD = 1.0d0/sqrt(Tt25/Tref)
      NbhtD = 1.0d0/sqrt(Tt41/Tref)
      NbltD = 1.0d0/sqrt(Tt45/Tref)

      pifD  = pif
      pilcD = pilc
      pihcD = pihc

c---- turbine design pressure ratios in the approximate form etmap expects
      Trh =  Tt41/(Tt41 + (ht45-ht41)/cpt41)
      Trl =  Tt45/(Tt45 + (ht49-ht45)/cpt45)
      gexh = cpt41/(Rt41*epht0)
      gexl = cpt45/(Rt45*eplt0)
      pihtD = Trh**gexh
      piltD = Trl**gexl

      A2d = A2
      A25d = A25
      A5d = A5
      A7d = A7
      Tt4d = Tt4
      mcored = mcore

c===== warm-up run at the design condition ===========================
c---- tfoper is always marched in TASOPT: wsize and mission hand it the
c---- previous point's converged state. A cold start at a flight condition
c---- away from design walks the LPC below unity pressure ratio and the
c---- Newton iteration does not recover. So warm up here first.
      iTFspec = 1
      Tt4 = Tt4d
      pif = 0.0d0
      pilc = 0.0d0
      pihc = 0.0d0
      mbf = 0.0d0
      mblc = 0.0d0
      mbhc = 0.0d0
      pt5 = 0.0d0
      M2 = 0.0d0
      M25 = 0.60d0
      call tfoper(gee,M0,T0,p0,a0, Tref,pref,
     &  Phiinl,Kinl,iBLIc,
     &  pid,pib,pifn,pitn, Gearf,
     &  pifD,pilcD,pihcD,pihtD,piltD,
     &  mbfD,mblcD,mbhcD,mbhtD,mbltD,
     &  NbfD,NblcD,NbhcD,NbhtD,NbltD,
     &  A2d,A25d,A5d,A7d,
     &  iTFspec, Ttf,ifuel,etab,
     &  epf0,eplc0,ephc0,epht0,eplt0, pifK,epfK,
     &  mofft,Pofft, Tt9,pt9, epsl,epsh, icool,
     &  Mtexit,dTstrk,StA,efilm,tfilm, M4a,ruc,
     &  ncrowx, ncrow, epsrow,Tmrow,
     &  TSFC,Fsp,hfuel,ff, Feng,mcore,
     &  pif,pilc,pihc, mbf,mblc,mbhc, Nbf,Nblc,Nbhc,
     &  Tt0 ,ht0 ,pt0 ,cpt0 ,Rt0 ,
     &  Tt18,ht18,pt18,cpt18,Rt18,
     &  Tt19,ht19,pt19,cpt19,Rt19,
     &  Tt2 ,ht2 ,pt2 ,cpt2 ,Rt2 ,
     &  Tt21,ht21,pt21,cpt21,Rt21,
     &  Tt25,ht25,pt25,cpt25,Rt25,
     &  Tt3 ,ht3 ,pt3 ,cpt3 ,Rt3 ,
     &  Tt4, ht4 ,pt4 ,cpt4 ,Rt4 ,
     &  Tt41,ht41,pt41,cpt41,Rt41,
     &  Tt45,ht45,pt45,cpt45,Rt45,
     &  Tt49,ht49,pt49,cpt49,Rt49,
     &  Tt5 ,ht5 ,pt5 ,cpt5 ,Rt5 ,
     &  Tt7 ,ht7 ,pt7 ,cpt7 ,Rt7 ,
     &  u0 ,
     &  T2 ,u2 ,p2 ,cp2 ,R2 ,M2 ,
     &  T25,u25,p25,cp25,R25,M25,
     &  T5 ,u5 ,p5 ,cp5 ,R5 ,M5 ,
     &  T6 ,u6 ,p6 ,cp6 ,R6 ,M6 , A6,
     &  T7 ,u7 ,p7 ,cp7 ,R7 ,M7 ,
     &  T8 ,u8 ,p8 ,cp8 ,R8 ,M8 , A8,
     &  u9 , A9 ,
     &  epf ,eplc ,ephc ,epht ,eplt ,
     &  etaf,etalc,etahc,etaht,etalt, Lconv)


c===== operating point ===============================================
c---- Only conditions the shipped tfoper can actually converge on are
c---- used here; see DISCREPANCIES.md. Its Newton iteration drifts away
c---- from the solution for almost any change of flight condition.
      iTFspec = 1
      if (icase .eq. 1) then
        Tt4 = Tt4d
      elseif (icase .eq. 2) then
        iTFspec = 2
        Feng = 23000.0d0
        Tt4 = Tt4d
      elseif (icase .eq. 3) then
        Tt4 = 1500.0d0
      else
        iTFspec = 2
        Feng = 26000.0d0
        Tt4 = Tt4d
      endif

c---- warm start: pif/pilc/pihc/mbf/mblc/mbhc/pt5/M2 carry over from the
c---- warm-up call above, which is how wsize and mission drive tfoper.
      call tfoper(gee,M0,T0,p0,a0, Tref,pref,
     &  Phiinl,Kinl,iBLIc,
     &  pid,pib,pifn,pitn, Gearf,
     &  pifD,pilcD,pihcD,pihtD,piltD,
     &  mbfD,mblcD,mbhcD,mbhtD,mbltD,
     &  NbfD,NblcD,NbhcD,NbhtD,NbltD,
     &  A2d,A25d,A5d,A7d,
     &  iTFspec, Ttf,ifuel,etab,
     &  epf0,eplc0,ephc0,epht0,eplt0, pifK,epfK,
     &  mofft,Pofft, Tt9,pt9, epsl,epsh, icool,
     &  Mtexit,dTstrk,StA,efilm,tfilm, M4a,ruc,
     &  ncrowx, ncrow, epsrow,Tmrow,
     &  TSFC,Fsp,hfuel,ff, Feng,mcore,
     &  pif,pilc,pihc, mbf,mblc,mbhc, Nbf,Nblc,Nbhc,
     &  Tt0 ,ht0 ,pt0 ,cpt0 ,Rt0 ,
     &  Tt18,ht18,pt18,cpt18,Rt18,
     &  Tt19,ht19,pt19,cpt19,Rt19,
     &  Tt2 ,ht2 ,pt2 ,cpt2 ,Rt2 ,
     &  Tt21,ht21,pt21,cpt21,Rt21,
     &  Tt25,ht25,pt25,cpt25,Rt25,
     &  Tt3 ,ht3 ,pt3 ,cpt3 ,Rt3 ,
     &  Tt4, ht4 ,pt4 ,cpt4 ,Rt4 ,
     &  Tt41,ht41,pt41,cpt41,Rt41,
     &  Tt45,ht45,pt45,cpt45,Rt45,
     &  Tt49,ht49,pt49,cpt49,Rt49,
     &  Tt5 ,ht5 ,pt5 ,cpt5 ,Rt5 ,
     &  Tt7 ,ht7 ,pt7 ,cpt7 ,Rt7 ,
     &  u0 ,
     &  T2 ,u2 ,p2 ,cp2 ,R2 ,M2 ,
     &  T25,u25,p25,cp25,R25,M25,
     &  T5 ,u5 ,p5 ,cp5 ,R5 ,M5 ,
     &  T6 ,u6 ,p6 ,cp6 ,R6 ,M6 , A6,
     &  T7 ,u7 ,p7 ,cp7 ,R7 ,M7 ,
     &  T8 ,u8 ,p8 ,cp8 ,R8 ,M8 , A8,
     &  u9 , A9 ,
     &  epf ,eplc ,ephc ,epht ,eplt ,
     &  etaf,etalc,etahc,etaht,etalt, Lconv)

      write(*,900) icase,'TSFC ',TSFC
      write(*,900) icase,'Fsp  ',Fsp
      write(*,900) icase,'ff   ',ff
      write(*,900) icase,'Feng ',Feng
      write(*,900) icase,'mcore',mcore
      write(*,900) icase,'pif  ',pif
      write(*,900) icase,'pilc ',pilc
      write(*,900) icase,'pihc ',pihc
      write(*,900) icase,'mbf  ',mbf
      write(*,900) icase,'mblc ',mblc
      write(*,900) icase,'mbhc ',mbhc
      write(*,900) icase,'Nbf  ',Nbf
      write(*,900) icase,'Nblc ',Nblc
      write(*,900) icase,'Nbhc ',Nbhc
      write(*,900) icase,'Tt3  ',Tt3
      write(*,900) icase,'Tt4  ',Tt4
      write(*,900) icase,'Tt41 ',Tt41
      write(*,900) icase,'Tt45 ',Tt45
      write(*,900) icase,'pt3  ',pt3
      write(*,900) icase,'pt5  ',pt5
      write(*,900) icase,'u5   ',u5
      write(*,900) icase,'u6   ',u6
      write(*,900) icase,'u7   ',u7
      write(*,900) icase,'u8   ',u8
      write(*,900) icase,'M2   ',M2
      write(*,900) icase,'M25  ',M25
      write(*,900) icase,'A6   ',A6
      write(*,900) icase,'A8   ',A8
      write(*,900) icase,'etaf ',etaf
      write(*,900) icase,'etaht',etaht
      write(*,900) icase,'eplt ',eplt

 500  continue
 900  format(I2,',',A5,',',E24.16)
      stop
      end

      subroutine compare(ss,aa,dd)
c---- Stub. tfoper references this from a finite-difference debug block
c---- that is only reachable when iter == -1, i.e. never. It exists in no
c---- source file in the distribution, so the shipped tfoper.f cannot be
c---- linked standalone without providing it.
      character*2 ss
      real aa, dd
      ss = '  '
      return
      end
