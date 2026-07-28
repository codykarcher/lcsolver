      program drv_tfcalc
c---- Size a turbofan through tfcalc, then run it off design through tfcalc.
c---- Case 1: sizing (icall=0), no cooling.
c---- Case 2: sizing with cooling (icool=2) and offtakes.
c---- Case 3: size, then off-design at specified Tt4 (icall=1), warm start.
c---- Case 4: size, then off-design at specified thrust (icall=2).
      implicit real (a-h,l-z)
      include 'index.inc'
      include 'constants.inc'
      integer pari(iitotal), icase, i, icall, icool, initeng
      integer ichoke5, ichoke7, ip
      real parg(igtotal), para(iatotal), pare(ietotal)

      pi  = 3.1415926535897932384626
      gee = 9.81
      cpSL = 1004.0
      gamSL = 1.4
      Tref = 288.2
      pref = 101320.0

      write(*,'(A)') 'case,name,value'
      do 500 icase = 1, 4

      do i = 1, iitotal
        pari(i) = 0
      enddo
      do i = 1, igtotal
        parg(i) = 0.0
      enddo
      do i = 1, iatotal
        para(i) = 0.0
      enddo
      do i = 1, ietotal
        pare(i) = 0.0
      enddo

      pari(iifuel) = 24
      pari(iiBLIc) = 0
      parg(igGearf) = 1.0
      parg(igneng)  = 2.0
      parg(igS)     = 124.0
      parg(igTmetal) = 1200.0
      parg(igHTRf)  = 0.30
      parg(igHTRlc) = 0.60
      parg(igHTRhc) = 0.80
      parg(igWpay)  = 1.7e5
      parg(igWMTO)  = 7.5e5

      pare(ieTfuel) = 280.0
      pare(ieTt4)   = 1450.0
      pare(ieBPR)   = 5.1
      pare(iepif)   = 1.685
      pare(iepilc)  = 1.935
      pare(iepihc)  = 9.369
      pare(iepid)   = 0.998
      pare(iepib)   = 0.94
      pare(iepifn)  = 0.98
      pare(iepitn)  = 0.989
      pare(ieepolf)  = 0.8948
      pare(ieepollc) = 0.88
      pare(ieepolhc) = 0.87
      pare(ieepolht) = 0.889
      pare(ieepollt) = 0.899
      pare(ieetab)  = 0.985
      pare(iepifK)  = 1.685
      pare(ieepfK)  = 0.0
      pare(ieM2)    = 0.60
      pare(ieM25)   = 0.60
      pare(ieM0)    = 0.80
      pare(ieT0)    = 219.43
      pare(iep0)    = 23842.0
      pare(iea0)    = sqrt(1.4*287.0*219.43)
      pare(ierho0)  = 23842.0/(287.0*219.43)
      pare(ieu0)    = 0.80*sqrt(1.4*287.0*219.43)
      pare(ieFe)    = 25000.0
      pare(iedTstrk) = 200.0
      pare(ieStA)    = 0.09
      pare(ieMtexit) = 1.0
      pare(ieM4a)    = 0.9
      pare(ieruc)    = 0.9
      pare(ieefilm)  = 0.7
      pare(ietfilm)  = 0.3
      pare(ieepsl)   = 0.0
      pare(ieepsh)   = 0.0
      pare(ieTt9)    = 300.0
      pare(iept9)    = 30000.0

      icool = 0
      if (icase .eq. 2) then
        icool = 2
        parg(igmofWMTO) = 0.5/7.5e5
        parg(igPofWMTO) = 60000.0/7.5e5
      endif

      ip = ipcruise1
      icall = 0
      initeng = 0
      call tfcalc(pari,parg,para,pare, ip, icall,icool,initeng,
     &            ichoke5,ichoke7)

      if (icase .ge. 3) then
c------ now run it off design, warm-started from the design state
        initeng = 1
        if (icase .eq. 3) then
          icall = 1
          pare(ieTt4) = 1450.0
        else
          icall = 2
          pare(ieFe) = 23000.0
        endif
        call tfcalc(pari,parg,para,pare, ip, icall,icool,initeng,
     &              ichoke5,ichoke7)
      endif

      write(*,900) icase,'TSFC ',pare(ieTSFC)
      write(*,900) icase,'Fsp  ',pare(ieFsp)
      write(*,900) icase,'ff   ',pare(ieff)
      write(*,900) icase,'mcore',pare(iemcore)
      write(*,900) icase,'Fe   ',pare(ieFe)
      write(*,900) icase,'Tt4  ',pare(ieTt4)
      write(*,900) icase,'BPR  ',pare(ieBPR)
      write(*,900) icase,'A2   ',pare(ieA2)
      write(*,900) icase,'A25  ',pare(ieA25)
      write(*,900) icase,'A5   ',pare(ieA5)
      write(*,900) icase,'A7   ',pare(ieA7)
      write(*,900) icase,'mbfD ',pare(iembfD)
      write(*,900) icase,'mblcD',pare(iemblcD)
      write(*,900) icase,'mbhcD',pare(iembhcD)
      write(*,900) icase,'mbhtD',pare(iembhtD)
      write(*,900) icase,'mbltD',pare(iembltD)
      write(*,900) icase,'NbfD ',pare(ieNbfD)
      write(*,900) icase,'pihtD',pare(iepihtD)
      write(*,900) icase,'piltD',pare(iepiltD)
      write(*,900) icase,'mbf  ',pare(iembf)
      write(*,900) icase,'mblc ',pare(iemblc)
      write(*,900) icase,'mbhc ',pare(iembhc)
      write(*,900) icase,'Nf   ',pare(ieNf)
      write(*,900) icase,'N1   ',pare(ieN1)
      write(*,900) icase,'N2   ',pare(ieN2)
      write(*,900) icase,'pif  ',pare(iepif)
      write(*,900) icase,'pilc ',pare(iepilc)
      write(*,900) icase,'pihc ',pare(iepihc)
      write(*,900) icase,'Tt3  ',pare(ieTt3)
      write(*,900) icase,'pt3  ',pare(iept3)
      write(*,900) icase,'Tt41 ',pare(ieTt41)
      write(*,900) icase,'Tt45 ',pare(ieTt45)
      write(*,900) icase,'dfan ',parg(igdfan)
      write(*,900) icase,'dlcom',parg(igdlcomp)
      write(*,900) icase,'dhcom',parg(igdhcomp)
      write(*,900) icase,'fc   ',pare(iefc)
      write(*,900) icase,'Phiin',pare(iePhiinl)

 500  continue
 900  format(I2,',',A5,',',E24.16)
      stop
      end

      subroutine compare(ss,aa,dd)
      character*2 ss
      real aa, dd
      ss = '  '
      return
      end
