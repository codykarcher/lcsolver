
      subroutine wsize(pari,parg,parm,para,pare,
     &                 iterwmax,wrlx1,wrlx2,wrlx3,
     &                 initwgt,initeng,iairf,
     &                 ichoke5,ichoke7,
     &                 Litprint,Lconv)
c=====================================================================
c     Sizes aircraft and calculates performance for design mission
c     
c     Input:
c      pari(.)    flags
c      iterwmax   max number of weight/sizing iterations
c      wrlx1      under-relaxation factor for   initial weight iterations
c      wrlx2      under-relaxation factor after initial weight iterations
c      wrlx3      under-relaxation factor near max iteration limit
c      initeng    0 = initialize engine states for engine calcs
c                 1 = use existing states to start engine calcs
c      iairf      index of airfoil database to use
c      Litprint   T = print weight-iteration history
c
c     Input/Output:
c      parg(.)   geometry parameters
c      parm(.)   mission  parameters
c      para(.p)  aero     parameters for points p=1..iptotal
c      pare(.p)  engine   parameters for points p=1..iptotal
c
c     Output:
c      ichoke5(p)  0 = core nozzle unchoked, 1 = core nozzle choked
c      ichoke7(p)  0 = fan  nozzle unchoked, 1 = fan  nozzle choked
c      Lconv       T = convergence successful
c=====================================================================
      implicit real (a-h,l-z)
c
      include 'index.inc'
      integer pari(iitotal)
      real parg(igtotal), 
     &     parm(imtotal),
     &     para(iatotal,iptotal),
     &     pare(ietotal,iptotal)
      integer iterwmax,initeng
      integer ichoke5(iptotal),
     &        ichoke7(iptotal)
      logical Litprint,Lconv
c
c
      include 'airf.inc'
      include 'constants.inc'

      integer ncrow, icrow

c---- arrays for calling mcool
      real Tmrow(ncrowx),
     &     epsrow(ncrowx),
     &     epsrow_Tt3(ncrowx),
     &     epsrow_Tt4(ncrowx),
     &     epsrow_Trr(ncrowx)

      logical Lengwrt, Lprint, Ldebug
      real KAfTE

      integer lu

      include 'time.inc'

      common /com_ip/ ip

c---- convergence tolerance, fractional weight change between iterations
      data tolerw / 1.0e-9 /
c      data tolerw / 1.0e-10 /
c      data tolerw / 1.0e-14 /


      Lengwrt = .false.   ! don't write engine parameters below
c      Lengwrt = .true.    ! write engine parameters below (for debugging)

      Ldebug = .false.
c      Ldebug = .true.

      if(Ldebug) write(*,*) 'entering WSIZE...'

c---- INSTRUMENTATION: dump the complete input state on entry.
c---- Not part of TASOPT -- see fortran_ref/wsize_instrumented.f.
      open(86,file='wsize_in.txt',status='unknown')
      write(86,'(i6)') (pari(k), k=1,iitotal)
      write(86,'(e26.18)') (parg(k), k=1,igtotal)
      write(86,'(e26.18)') (parm(k), k=1,imtotal)
      do k = 1, iatotal
        write(86,'(e26.18)') (para(k,ipx), ipx=1,iptotal)
      enddo
      do k = 1, ietotal
        write(86,'(e26.18)') (pare(k,ipx), ipx=1,iptotal)
      enddo
      write(86,'(i6)') iterwmax, initwgt, initeng, iairf
      write(86,'(e26.18)') wrlx1, wrlx2, wrlx3
      close(86)

      errw = 1.0

      ifuel   = pari(iifuel  )
      ifwcen  = pari(iifwcen )
      iwplan  = pari(iiwplan )
      iengloc = pari(iiengloc)
      iengwgt = pari(iiengwgt)
      iBLIc   = pari(iiBLIc  )
      ifclose = pari(iifclose)
      iHTsize = pari(iiHTsize)
      iVTsize = pari(iiVTsize)
      ixwmove = pari(iixwmove)

c---- calculate fuselage BL development for start of cruise point
      ip = ipcruise1
c      call tset(time0)
      call fusebl(pari,parg,para(1,ip))
c      call tadd(time0,t_fusebl)

c---- assume K.E., dissipation, drag areas will be the same for all points
      KAfTE   = para(iaKAfTE,ip)
      DAfsurf = para(iaDAfsurf,ip)
      DAfwake = para(iaDAfwake,ip)
      PAfinf  = para(iaPAfinf,ip)
      do ip = 1, iptotal
        para(iaKAfTE,ip)   = KAfTE
        para(iaDAfsurf,ip) = DAfsurf
        para(iaDAfwake,ip) = DAfwake
        para(iaPAfinf,ip)  = PAfinf
      enddo

c------------------------------------------------------------------------
c---- set specified quantities which are fixed during weight iteration loop

c---- parameters for sizing mission
      Rangetot = parm(imRange)
      Wpay     = parm(imWpay)

c---- store design-mission parameters in geometry array
      parg(igRange) = Rangetot
      parg(igWpay ) = Wpay

c---- fixed weight and location
      Wfix = parg(igWfix)
      xfix = parg(igxfix)

c---- weight fractions 
      fapu  = parg(igfapu )
      fpadd = parg(igfpadd)
      fseat = parg(igfseat)
      feadd = parg(igfeadd)
      fnace = parg(igfnace)
      fhadd = parg(igfhadd)
      fvadd = parg(igfvadd)
      fwadd = parg(igfflap)
     &      + parg(igfslat)
     &      + parg(igfaile)
     &      + parg(igflete)
     &      + parg(igfribs)
     &      + parg(igfspoi)
     &      + parg(igfwatt)

      fstring = parg(igfstring)
      fframe  = parg(igfframe )
      ffadd   = parg(igffadd  )

      fpylon  = parg(igfpylon)

      fhpesys = parg(igfhpesys)
      flgnose = parg(igflgnose)
      flgmain = parg(igflgmain)

      freserve = parg(igfreserve)

c---- fuselage lift carryover loss, tip lift loss factors
      fLo = parg(igfLo)
      fLt = parg(igfLt)

c---- fuselage dimensions and coordinates
      Rfuse  = parg(igRfuse)
      dRfuse = parg(igdRfuse)
      wfb    = parg(igwfb  )
      nfweb  = parg(ignfweb)
      hfloor = parg(ighfloor)
      xnose   = parg(igxnose)
      xend    = parg(igxend)
      xshell1 = parg(igxshell1)
      xshell2 = parg(igxshell2)
      xconend = parg(igxconend)
      xwbox   = parg(igxwbox)
      xhbox   = parg(igxhbox)
      xvbox   = parg(igxvbox)
      xapu    = parg(igxapu)
      xeng    = parg(igxeng)

c---- payload-proportional weights
      Wapu  = Wpay*fapu
      Wpadd = Wpay*fpadd
      Wseat = Wpay*fseat

c---- window and insulation densities per length and per area
      Wpwindow = parg(igWpwindow)
      Wppinsul = parg(igWppinsul)
      Wppfloor = parg(igWppfloor)

c---- fuselage-bending inertial relief factors
      rMh = parg(igrMh)
      rMv = parg(igrMv)

c---- tail CL's at structural sizing cases
      CLhmax = parg(igCLhmax)
      CLvmax = parg(igCLvmax)

c---- wing break, wing tip taper ratios
      lambdas = parg(iglambdas)
      lambdat = parg(iglambdat)

c---- tail surface taper ratios (no inner panel, so lambdas=1)
      lambdahs = 1.0
      lambdah = parg(iglambdah)
      lambdavs = 1.0
      lambdav = parg(iglambdav)

c---- tailcone taper ratio
      lambdac = parg(iglambdac)

c---- wing geometry parameters
      sweep = parg(igsweep)
      wbox  = parg(igwbox )
      hboxo = parg(ighboxo)
      hboxs = parg(ighboxs)
      rh    = parg(igrh)
      AR    = parg(igAR)
      bo    = parg(igbo)
      etas  = parg(igetas)
      Xaxis = parg(igXaxis)

c---- tail geometry parameters
      sweeph = parg(igsweeph)
      wboxh  = parg(igwboxh)
      hboxh  = parg(ighboxh)
      rhh    = parg(igrhh)
      ARh    = parg(igARh)
      boh = parg(igboh)

      sweepv = parg(igsweepv)
      wboxv  = parg(igwboxv)
      hboxv  = parg(ighboxv)
      rhv    = parg(igrhv)
      ARv    = parg(igARv)
      bov = parg(igbov)

c---- number of vertical tails
      nvtail = parg(ignvtail)

c---- strut vertical base height, h/c, strut shell t/h
      zs     = parg(igzs)
      hstrut = parg(ighstrut)
      tohstrut = 0.05

c---- assume no struts on tails
      zsh = 0.0
      zsv = 0.0

c---- max g load factors for wing, fuselage
      Nlift = parg(igNlift)
      Nland = parg(igNland)

c---- never-exceed dynamic pressure for sizing tail structure
      Vne = parg(igVne)
      qne = 0.5*rhoSL*Vne**2

c---- wingbox stresses and densities
      sigcap  = parg(igsigcap ) * parg(igsigfac)
      tauweb  = parg(igtauweb ) * parg(igsigfac)
      rhoweb  = parg(igrhoweb )
      rhocap  = parg(igrhocap )

c---- fuselage stresses and densities
      sigskin = parg(igsigskin) * parg(igsigfac)
      sigbend = parg(igsigbend) * parg(igsigfac)
      rhoskin = parg(igrhoskin)
      rhobend = parg(igrhobend)

c---- fuselage shell bending/skin modulus ratio
      rEshell = parg(igrEshell)

c---- strut stress and density
      sigstrut = parg(igsigstrut) * parg(igsigfac)
      rhostrut = parg(igrhostrut)

c---- assume tail stresses and densities are same as wing's (keeps it simpler)
      sigcaph = sigcap
      tauwebh = tauweb
      rhowebh = rhoweb
      rhocaph = rhocap

      sigcapv = sigcap
      tauwebv = tauweb
      rhowebv = rhoweb
      rhocapv = rhocap

c---- number of engines, y-position of outermost engine
      neng = parg(igneng)
      yeng = parg(igyeng)

c---- fan hub/tip ratio
      HTRf = parg(igHTRf)

c---- nacelle wetted area / fan area ratio
      rSnace = parg(igrSnace)

c---- set cruise-altitude atmospheric conditions
      ip = ipcruise1
      altkm = para(iaalt,ip)/1000.0
      call atmos(altkm, T0,p0,rho0,a0,mu0)
      Mach = para(iaMach,ip)
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      pare(ieM0  ,ip)   = Mach
      pare(ieu0  ,ip)   = Mach*a0
      para(iaReunit,ip) = Mach*a0 * rho0/mu0

c---- set takeoff-altitude atmospheric conditions
      ip = iprotate
      altkm = para(iaalt,ip)/1000.0
      call atmos(altkm, T0,p0,rho0,a0,mu0)
      Mach = 0.25
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      pare(ieM0  ,ip)   = Mach
      pare(ieu0  ,ip)   = Mach*a0
      para(iaReunit,ip) = Mach*a0 * rho0/mu0

c==========================================================================
c---- initial-guess section, to allow starting weight iteration loop
c-     (none of these guesses affect the final result, provided it converges)
      if(initwgt .eq. 0) then

      if(Ldebug) write(*,*) 'initial guesses...'

      Whtail = 0.05 * Wpay / parg(igsigfac)
      Wvtail = 0.05 * Wpay / parg(igsigfac)
      Wwing  = 0.5  * Wpay / parg(igsigfac)
      Wstrut = 0.0  * Wpay / parg(igsigfac)
      Weng   = 0.3  * Wpay
      feng   = 0.08

      dxWhtail = 0.
      dxWvtail = 0.

c---- wing panel weights and moments (estimate span first)
      ip = ipcruise1
      W = 5.0*Wpay
      S = W / (0.5*pare(ierho0,ip)*pare(ieu0,ip)**2 * para(iaCL,ip))
      b = sqrt(S*parg(igAR))
      bs = b*etas
      Winn = 0.15 * Wpay / parg(igsigfac)
      Wout = 0.05 * Wpay / parg(igsigfac)
      dyWinn = Winn*0.30*(0.5*(bs-bo))
      dyWout = Wout*0.25*(0.5*(b -bs))

      parg(igWhtail) = Whtail
      parg(igWvtail) = Wvtail
      parg(igWwing ) = Wwing
      parg(igWstrut) = Wstrut
      parg(igWeng  ) = Weng
      parg(igWinn  ) = Winn
      parg(igWout  ) = Wout
      parg(igdxWhtail) = dxWhtail
      parg(igdxWvtail) = dxWvtail
      parg(igdyWinn) = dyWinn
      parg(igdyWout) = dyWout

c---- wing centroid x-offset from wingbox
      call surfdx(b,bs,bo,lambdat,lambdas,sweep,
     &            dxwing, macco)
      xwing = xwbox + dxwing
      parg(igxwing) = xwing

c---- tail area centroid locations (assume no offset from sweep initially)
      parg(igxhtail) = xhbox
      parg(igxvtail) = xvbox

c---- center wing box chord extent for fuselage weight calcs (small effect)
      cbox = 0.

c---- nacelle, fan duct, core cowl lengths Re calculations
      parg(iglnace)  = 0.5*S/b

c---- nacelle Awet/S
      fSnace = 0.2
      parg(igfSnace) = fSnace

c---- estimate fuel fraction from Breguet
      DoL = 1.0/18.0
      TSFC = 1.0/7000.0
      V = pare(ieu0,ipcruise1)
      ffburn = (1.0 - exp(-Rangetot*DoL*TSFC/V))
      ffburn = min( ffburn , 0.8/(1.0+freserve) )

c---- mission-point fuel fractions
      ffuelb = ffburn*(1.0  + freserve)  ! start of climb
      ffuelc = ffburn*(0.90 + freserve)  ! start of cruise
      ffueld = ffburn*(0.02 + freserve)  ! start of descent
      ffuele = ffburn*(0.0  + freserve)  ! end of descent (landing)

c---- max fuel fraction is at start of climb
      ffuel = ffuelb

c---- clear climb angles, to force initial guesses
      do ip = 1, iptotal
        para(iagamV,ip) = 0.
      enddo

c---- put initial-guess weight fractions in mission-point array
      para(iafracW,ipstatic ) = 1.0
      para(iafracW,iprotate ) = 1.0
      para(iafracW,iptakeoff) = 1.0
      para(iafracW,ipcutback) = 1.0

      do ip = ipclimb1, ipclimbn
        frac = float(ip      -ipclimb1)
     &       / float(ipclimbn-ipclimb1)
        ffp = ffuelb*(1.0-frac) + ffuelc*frac
        para(iafracW,ip) = 1.0 - ffuel + ffp
      enddo

      do ip = ipcruise1, ipcruisen
        frac = float(ip       -ipcruise1)
     &       / float(ipcruisen-ipcruise1)
        ffp = ffuelc*(1.0-frac) + ffueld*frac
        para(iafracW,ip) = 1.0 - ffuel + ffp
      enddo

      do ip = ipdescent1, ipdescentn
        frac = float(ip        -ipdescent1)
     &       / float(ipdescentn-ipdescent1)
        ffp = ffueld*(1.0-frac) + ffuele*frac
        para(iafracW,ip) = 1.0 - ffuel + ffp
      enddo

c---- initial tail info needed for initial fuselage bending and torsion material
      Sh = (2.0*Wpay) / (qne*CLhmax)
      Sv = (2.0*Wpay) / (qne*CLvmax)
      bv = sqrt(Sv*ARv)

      parg(igSh) = Sh
      parg(igSv) = Sv

c---- wing and tail total pitching moments (including sweep)
      do ip = 1, iptotal
        para(iaCMw0,ip) = 0.
        para(iaCMw1,ip) = 0.
        para(iaCMh0,ip) = 0.
        para(iaCMh1,ip) = 0.
        para(iaCLh,ip) = 0.
      enddo

c---- initial cruise-climb angle gamVcr needed to estimate end-of cruise altitude,
c-     to set initial cabin-pressure deltap for sizing fuselage shell
      LoD = 18.0
      gamVcr = 0.0002
      para(iaCD  ,ipcruise1) = para(iaCL,ipcruise1) / LoD
      para(iagamV,ipcruise1) = gamVcr
      
c---- pressure and altitude at start of cruise 
c-     (this won't change -- it's needed for end of cruise stuff below)
      Mach = para(iaMach,ipcruise1)
      p0c  = pare(iep0  ,ipcruise1)
      altc = para(iaalt ,ipcruise1)

c---- guess pressure p0d at end-of-cruise (scales with weight),
c-     for initial cabin delta(p)
      p0d  = p0c * (1.0-ffuel+ffueld)/(1.0-ffuel+ffuelc)
      pare(iep0,ipcruisen) = p0d

c---- guess for single-engine thrust, speed, fan area, for engine-out VT sizing
      pare(ieFe,iprotate) = 2.0*Wpay / neng
      pare(ieu0,iprotate) = 70.0
      Afan = 3.0e-5 * Wpay / neng
      parg(igdfan) = sqrt( Afan*4.0/pi )


c---- guess for fan-face Mach numbers, for nacelle CD calculation
      M2des = pare(ieM2,ipcruise1)
      do ip = ipstatic, ipcruisen
        pare(ieM2,ip) = M2des
      enddo
      do ip = ipdescent1, ipdescentn
        pare(ieM2,ip) = 0.8 * M2des
      enddo

c---- calculate initial guesses for cooling mass flow ratios epsrow(.)
      ip = iprotate
cc    cpc = 1025.0   ! average compressor cp
      cpc = 1080.0   ! average compressor cp
      cp4 = 1340.0
      Rgc = 288.0    ! average compressor R
      Rg4 = 288.0
      M0to  = pare(ieu0,ip)/pare(iea0,ip)
      T0to  = pare(ieT0,ip)
      epolhc= pare(ieepolhc,ip)
      OPRto = pare(iepilc,ipcruise1)*pare(iepihc,ipcruise1)
      Tt4to = pare(ieTt4,ip)
      dTstrk= pare(iedTstrk,ip)
      Mtexit= pare(ieMtexit,ip)
      efilm = pare(ieefilm,ip)
      tfilm = pare(ietfilm,ip)
      StA   = pare(ieStA,ip)
      do icrow = 1, ncrowx
        Tmrow(icrow) = parg(igTmetal)
      enddo

      Tt2to = T0to*(1.0 + 0.5*(gamSL-1.0) * M0to**2)
      Tt3to = Tt2to * OPRto**(Rgc/(epolhc*cpc))
      Trrat = 1.0 / (1.0 + 0.5*Rg4/(cp4-Rg4) * Mtexit**2)
c      call tset(time0)
      call mcool(ncrowx, ncrow,
     &           Tmrow, Tt3to,Tt4to,dTstrk, Trrat,
     &           efilm,tfilm,StA, 
     &           epsrow,epsrow_Tt3,epsrow_Tt4,epsrow_Trr)
      epstot = 0.
      do icrow = 1, ncrow
        epstot = epstot + epsrow(icrow)
      enddo
      fo = pare(iemofft,ip)/pare(iemcore,ip)
      fc = (1.0-fo)*epstot
c      call tadd(time0,t_mcool)

      if(fc .ge. 0.99) then
       write(*,*) 
       write(*,*) 'WSIZE: Excessive cooling flow'
       write(*,*) 'mcool/mcore =', fc
       write(*,*) 'Tt3 Tt4 Tmetal', Tt3to, Tt4to, parg(igTmetal)
       stop
      endif

      do jp = 1, iptotal
        pare(iefc,jp) = fc
        do icrow = 1, ncrowx
          pare(ieepsc1+icrow-1,jp) = epsrow(icrow)
          pare(ieTmet1+icrow-1,jp) = Tmrow(icrow)
        enddo
      enddo

c------------------------------------------------------------------
      else
c---- use current weights as initial guesses for sizing loop

      S = parg(igS)
      b = parg(igb)
      bs = parg(igbs)
      bo = parg(igbo)

      bh = parg(igbh)
      bv = parg(igbv)

      coh = parg(igcoh)
      cov = parg(igcov)

      cbox = parg(igco)*parg(igwbox)


      Whtail = parg(igWhtail)
      Wvtail = parg(igWvtail)
      Wwing  = parg(igWwing )
      Wstrut = parg(igWstrut)
      Weng   = parg(igWeng  )
      Winn   = parg(igWinn  )
      Wout   = parg(igWout  )
      dxWhtail = parg(igdxWhtail)
      dxWvtail = parg(igdxWvtail)
      dyWinn = parg(igdyWinn)
      dyWout = parg(igdyWout)

      WMTO  = parg(igWMTO)
      feng  = parg(igWeng)/WMTO
      ffuel = parg(igWfuel)/WMTO

      xwing  = parg(igxwing)
      dxwing = parg(igxwing) - parg(igxwbox)

      xhtail = parg(igxhtail)
      xvtail = parg(igxvtail)

      xhbox = parg(igxhbox)
      xvbox = parg(igxvbox)

      fSnace = parg(igfSnace)

      Sh = parg(igSh)
      Sv = parg(igSv)
      ARh = parg(igARh)
      ARv = parg(igARv)

      endif
c==========================================================================

c---- initialize previous-iteration weights (none yet)
      WMTO1 = 0.0  ! 1st-previous-iteration weight, for convergence criterion
      WMTO2 = 0.0  ! 2nd-previous-iteration weight, for convergence criterion
      WMTO3 = 0.0  ! 3rd-previous-iteration weight, for convergence criterion

c---- no convergence yet
      Lconv = .false.

c---- set these to zero for first-iteration info printout
      parg(igb) = 0.
      parg(igS) = 0.
      do ip = 1, iptotal
        ichoke5(ip) = 0
        ichoke7(ip) = 0
      enddo

c---- engine core mass flow will need to be initialized
c      do ip = 1, iptotal
c        pare(iemcore,ip) = 0.
c      enddo

      if(Ldebug) write(*,*) 'Starting weight iteration ...'

c==============================================================================
c---- top of weight-iteration loop
      do 100 iterw = 1, iterwmax

      if(initwgt .eq. 0) then
c----- current weight iteration started from initial guess, so be cautious
       itrlx = 5

      else
c----- current weight iteration started from previous converged solution
       itrlx = 2

      endif

      if(iterw .le. itrlx) then
c----- under-relax first nitrlx iterations
       rlx = wrlx1

      elseif(iterw .ge. (3*iterwmax)/4) then
c----- under-relax after 3/4'ths of max iterations  (quashes likely limit cycle)
       rlx = wrlx3
      
      else
c----- default is no under-relaxation for weight update
       rlx = wrlx2

      endif

c-------------------------------------------------------------------------
c---- fuselage sizing section

c---- max tail lifts at maneuver dynamic pressure qne
      Lhmax = qne*Sh*CLhmax
      Lvmax = qne*Sv*CLvmax/nvtail

c---- max deltap (fuselage overpressure) at end of cruise-climb, assumes p ~ W/S
      wcd = para(iafracW,ipcruisen)
     &     /para(iafracW,ipcruise1)
      deltap = parg(igpcabin) - pare(iep0,ipcruise1)*wcd
      parg(igdeltap) = deltap

c---- engine weight, if any, mounted on tailcone
      if(iengloc.eq.1) then
       Wengtail = 0.
      else
       Wengtail = parg(igWeng)
      endif

cc    Wfix  = parg(igWfix)
cc    Wpay  = parg(igWpay)
cc    Wpadd = parg(igWpay)*parg(igfpadd)
      Whtail = parg(igWhtail)
      Wvtail = parg(igWvtail)
      xhtail = parg(igxhtail)
      xvtail = parg(igxvtail)
      xwbox  = parg(igxwbox)
      xwing  = parg(igxwing)

      if(Ldebug) write(*,*) 'Calling FUSEW...'

c---- fuselage weight, with skin sized by max deltap, tail loads sized by qne
c      call tset(time0)
      Eskin = parg(igEcap)
      Ebend = Eskin*rEshell
      Gskin = Eskin * 0.5/(1.0+0.3)
      call fusew(gee,Nland,Wfix,Wpay,Wpadd,Wseat,Wapu,Wengtail,
     &           fstring,fframe,ffadd,deltap,
     &           Wpwindow,Wppinsul,Wppfloor,
     &           Whtail,Wvtail,rMh,rMv,Lhmax,Lvmax,
     &           bv,lambdav,nvtail,
     &           Rfuse,dRfuse,wfb,nfweb,lambdac,
     &           xnose,xshell1,xshell2,xconend,
     &           xhtail,xvtail,
     &           xwing,xwbox,cbox,
     &           xfix,xapu,xeng,
     &           hfloor,
     &           sigskin,sigbend, rhoskin,rhobend,
     &           Eskin,Ebend,Gskin,
     &           tskin, tcone, tfweb, tfloor, xhbend, xvbend,
     &           EIhshell,EIhbend,
     &           EIvshell,EIvbend,
     &           GJshell ,GJcone,
     &           Wshell, Wcone, Wwindow, Winsul, Wfloor, 
     &           Whbend, Wvbend,
     &           Wfuse,
     &          xWfuse,
     &           cabVol )
c      call tadd(time0,t_fusew)

      parg(igtskin )  = tskin
      parg(igtcone )  = tcone
      parg(igtfweb )  = tfweb
      parg(igtfloor)  = tfloor
      parg(igxhbend)  = xhbend
      parg(igxvbend)  = xvbend

      parg(igEIhshell) = EIhshell
      parg(igEIhbend ) = EIhbend 
      parg(igEIvshell) = EIvshell
      parg(igEIvbend ) = EIvbend 
      parg(igGJshell ) = GJshell 
      parg(igGJcone  ) = GJcone  

      parg(igWshell)  = Wshell
      parg(igWcone )  = Wcone
      parg(igWwindow) = Wwindow
      parg(igWinsul)  = Winsul
      parg(igWfloor)  = Wfloor

      parg(igWhbend)  = Whbend
      parg(igWvbend)  = Wvbend

      parg(igWfuse ) = Wfuse
      parg(igxWfuse) = xWfuse

      parg(igcabVol) = cabVol

c---- buoyancy weight needed to size wing and engine at cruise
      ip = ipcruise1
      rhocab = max( parg(igpcabin) , pare(iep0,ip) ) / (RSL*TSL)
      WbuoyCR = (rhocab - pare(ierho0,ip))*gee*cabVol

c-------------------------------------------------------------------------
c---- total max takeoff weight
c      WMTO = Wpay + Wfuse
c    &      + Wwing + Wstrut + Whtail + Wvtail
c    &      + Weng + Wfuel
c    &      + Whpesys + Wlgnose + Wlgmain

      if(iterw .eq. 1 .and. initwgt .eq. 0) then
       fsum = feng + ffuel
     &      + fhpesys + flgnose + flgmain
       WMTO = ( Wpay + Wfuse
     &        + Wwing + Wstrut + Whtail + Wvtail)
     &     / (1.0 - fsum)
       Weng    = WMTO*feng
       Wfuel   = WMTO*ffuel 
       Whpesys = WMTO*fhpesys
       Wlgnose = WMTO*flgnose
       Wlgmain = WMTO*flgmain
       parg(igWMTO ) = WMTO
       parg(igWeng ) = Weng    
       parg(igWfuel) = Wfuel   

      else
       call Wupdate0(parg,rlx,fsum)
       if(fsum .ge. 1.0) go to 110

       parm(imWTO)   = parg(igWMTO)
       parm(imWfuel) = parg(igWfuel)

       if(Ldebug) then
       write(*,'(1x,i3, 10f10.1, 2f10.6)') 1,
     &  parg(igWMTO)  ,
     &  parg(igWfuel)  ,
     &  parg(igWwing)  ,
     &  parg(igWstrut) ,
     &  parg(igWfuse)  ,
     &  parg(igWhtail) ,
     &  parg(igWvtail) ,
     &  parg(igWeng),
     &  para(iaalt,ipcruise1),
     &  para(iaalt,ipcruisen),
     &  para(iaCL,ipcruise1)/para(iaCD,ipcruise1),
     &  para(iaCL,ipcruisen)/para(iaCD,ipcruisen)
       endif

      endif

c---- this calculated WMTO is the design-mission WTO
      parm(imWTO) = parg(igWMTO)

c-------------------------------------------------------------------------
c---- convergence tests

c---- set max error from last 2 iterations 
c-    (prevents false convergence from "lucky" near-zero single change)
      WMTO = parg(igWMTO)
      errw1 = (WMTO-WMTO1)/WMTO
      errw2 = (WMTO-WMTO2)/WMTO
      errw3 = (WMTO-WMTO3)/WMTO

      errw = max( abs(errw1) , abs(errw2) , abs(errw3) )

      if(Litprint) then
        if(iterw.eq.1) then
          write(*,'(/1x,a,a,a,a,a)')
     & ' iterw     errW    ',
     & '      WMTO        Wfuel        Wfuse        Wwing         Weng',
     & '        span     area     HTarea   xwbox'
        endif

 4100   format(1x,i5, f14.10, 5f13.4, f9.3, f10.3, f10.3, 5f12.5)
        write(*,4100)
     &     iterw, errw1,
     &     parm(imWTO) *lb_N,
     &     parg(igWfuel)*lb_N,
     &     parg(igWfuse)*lb_N,
     &     parg(igWwing)*lb_N,
     &     parg(igWeng )*lb_N,
     &     parg(igb)*ft_m,
     &     parg(igS)*ft_m**2,
     &     parg(igSh)*ft_m**2,
     &     parg(igxwbox)*ft_m
c     &     para(iagamV,ipclimb1)*180.0/pi,
c     &     para(iagamV,ipclimbn)*180.0/pi
c     &     parg(igxwing)*ft_m,
c     &     dxwing*ft_m

c     &     'TO', ichoke5(iptakeoff),ichoke7(iptakeoff),
c     &     'B1', ichoke5(ipclimb1 ),ichoke7(ipclimb1 ),
c     &     'Bn', ichoke5(ipclimbn ),ichoke7(ipclimbn ),
c     &     'C1', ichoke5(ipcruise1),ichoke7(ipcruise1),
c     &     'Cn', ichoke5(ipcruisen),ichoke7(ipcruisen)

      endif

      if(errw .lt. tolerw) then
        Lconv = .true.
        go to 150
      endif

c----------------------------------------------------------------------------
c---- wing sizing section
      WMTO = parg(igWMTO)

c---- size wing area and chords at start-of-cruise
      ip = ipcruise1
      W = WMTO*para(iafracW,ip)
      CL = para(iaCL,ip)
      rho0 = pare(ierho0,ip)
      u0   = pare(ieu0  ,ip)
      qinf = 0.5*rho0*u0**2
      BW = W + WbuoyCR

      call wingsc(BW,CL,qinf,AR,etas,bo,lambdat,lambdas,
     &            S,b,bs,co)
      parg(igS  ) = S
      parg(igb  ) = b
      parg(igbs ) = bs
      parg(igco ) = co

c---- update center wing box chord for FUSEW on next cycle
      cbox = co*wbox

c---- wing centroid x-offset from wingbox
      call surfdx(b,bs,bo,lambdat,lambdas,sweep,
     &             dxwing,macco)
      xwing = xwbox + dxwing
      cma = macco * co
      parg(igxwing) = xwing
      parg(igcma) = cma

c---- set wing pitching moment constants
      ip = iptakeoff
      cmpo = para(iacmpo,ip)
      cmps = para(iacmps,ip)
      cmpt = para(iacmpt,ip)
      gammat = parg(iglambdat)*para(iarclt,ip)
      gammas = parg(iglambdas)*para(iarcls,ip)
      call surfcm(b,bs,bo, sweep, Xaxis,
     &            lambdat,lambdas,gammat,gammas,
     &            AR,fLo,fLt,cmpo,cmps,cmpt,
     &            CMw0,CMw1)
      do ip = ipstatic, ipclimb1
        para(iaCMw0,ip) = CMw0
        para(iaCMw1,ip) = CMw1
      enddo

      ip = ipcruise1
      cmpo = para(iacmpo,ip)
      cmps = para(iacmps,ip)
      cmpt = para(iacmpt,ip)
      gammat = parg(iglambdat)*para(iarclt,ip)
      gammas = parg(iglambdas)*para(iarcls,ip)
      call surfcm(b,bs,bo, sweep, Xaxis,
     &            lambdat,lambdas,gammat,gammas,
     &            AR,fLo,fLt,cmpo,cmps,cmpt,
     &            CMw0,CMw1)
      do ip = ipclimb1+1, ipdescentn-1
        para(iaCMw0,ip) = CMw0
        para(iaCMw1,ip) = CMw1
      enddo

      ip = ipdescentn
      cmpo = para(iacmpo,ip)
      cmps = para(iacmps,ip)
      cmpt = para(iacmpt,ip)
      gammat = parg(iglambdat)*para(iarclt,ip)
      gammas = parg(iglambdas)*para(iarcls,ip)
      call surfcm(b,bs,bo, sweep, Xaxis,
     &            lambdat,lambdas,gammat,gammas,
     &            AR,fLo,fLt,cmpo,cmps,cmpt,
     &            CMw0,CMw1)
      do ip = ipdescentn, ipdescentn
        para(iaCMw0,ip) = CMw0
        para(iaCMw1,ip) = CMw1
      enddo

c---- set wing center load po using cruise spanload cl(y)
      ip = ipcruise1
      gammat = parg(iglambdat)*para(iarclt,ip)
      gammas = parg(iglambdas)*para(iarcls,ip)
      Lhtail = WMTO * parg(igCLhNrat)*parg(igSh)/parg(igS)
      call wingpo(b,bs,bo,
     &             lambdat,lambdas,gammat,gammas,
     &             AR,Nlift,WMTO,Lhtail,fLo,fLt,
     &             po)

      if(iwplan.eq.1) then
c----- engines on wing, at ys=bs/2
       Weng1 = parg(igWeng)/neng
      else
c----- engines not mounted on wing
       Weng1 = 0.
      endif

      if(Ldebug) write(*,*) 'calling wing surfw...'

c---- size wing structure and weights
      Winn = parg(igWinn)
      Wout = parg(igWout)
      dyWinn = parg(igdyWinn)
      dyWout = parg(igdyWout)
      rhofuel = parg(igrhofuel)
      Ecap = parg(igEcap)
      Eweb = Ecap
      Gcap = Ecap*0.5/(1.0+0.3)
      Gweb = Ecap*0.5/(1.0+0.3)
      call surfw(gee,po, b,bs,bo,co,zs,
     &           lambdat,lambdas,gammat,gammas,
     &           Nlift,iwplan,Weng1,
     &           Winn,Wout,dyWinn,dyWout,
     &           sweep,wbox,hboxo,hboxs,rh, fLt,
     &           tauweb,sigcap,sigstrut,Ecap,Eweb,Gcap,Gweb,
     &           rhoweb,rhocap,rhostrut,rhofuel,
     &           Ss,Ms,tbwebs,tbcaps,EIcs,EIns,GJs,
     &           So,Mo,tbwebo,tbcapo,EIco,EIno,GJo,
     &           Astrut,lstrutp,cosLs,
     &           Wscen,Wsinn,Wsout,dxWsinn,dxWsout,dyWsinn,dyWsout,
     &           Wfcen,Wfinn,Wfout,dxWfinn,dxWfout,dyWfinn,dyWfout,
     &           Wweb,  Wcap,  Wstrut,
     &         dxWweb,dxWcap,dxWstrut )

      Wwing   = 2.0*(Wscen +  Wsinn +   Wsout)*(1.0+fwadd)
      dxWwing = 2.0*(       dxWsinn + dxWsout)*(1.0+fwadd)

      if(pari(iifwcen) .eq. 0) then
       Wfmax   = 2.0*(         Wfinn +   Wfout)
       dxWfmax = 2.0*(       dxWfinn + dxWfout)
      else
       Wfmax   = 2.0*(Wfcen +  Wfinn +   Wfout)
       dxWfmax = 2.0*(       dxWfinn + dxWfout)
      endif

      rfmax = parg(igWfuel)/Wfmax
      parg(igWwing) = Wwing * rlx    +   parg(igWwing)*(1.0-rlx)
      parg(igWfmax) = Wfmax
      parg(igdxWwing) = dxWwing
      parg(igdxWfuel) = dxWfmax*rfmax

      parg(igtbwebs) = tbwebs
      parg(igtbcaps) = tbcaps
      parg(igtbwebo) = tbwebo
      parg(igtbcapo) = tbcapo
      parg(igAstrut) = Astrut
      parg(igcosLs ) = cosLs 
      parg(igWweb  ) = Wweb  
      parg(igWcap  ) = Wcap  
      parg(igWstrut) = Wstrut
      parg(igSomax) = So
      parg(igMomax) = Mo
      parg(igSsmax) = Ss
      parg(igMsmax) = Ms
      parg(igEIco) = EIco
      parg(igEIcs) = EIcs
      parg(igEIno) = EIno
      parg(igEIns) = EIns
      parg(igGJo)  = GJo
      parg(igGJs)  = GJs


      parg(igWstrut) = Wstrut
      parg(igdxWstrut) = dxWstrut

c---- strut chord (perpendicular to strut)
      cstrut = sqrt(0.5*Astrut/(tohstrut*hstrut))
      Sstrut = 2.0*cstrut*lstrutp
      parg(igcstrut) = cstrut
      parg(igSstrut) = Sstrut

c---- individual panel weights
      Wfuel = parg(igWfuel)
      rfmax = Wfuel/Wfmax
      Winn = Wsinn*(1.0+fwadd) + rfmax*Wfinn
      Wout = Wsout*(1.0+fwadd) + rfmax*Wfout
      dyWinn = dyWsinn*(1.0+fwadd) + rfmax*dyWfinn
      dyWout = dyWsout*(1.0+fwadd) + rfmax*dyWfout

      parg(igWinn) = Winn
      parg(igWout) = Wout
      parg(igdyWinn) = dyWinn
      parg(igdyWout) = dyWout

c----------------------------------------------------------------------------
c---- tail sizing section

      if(Ldebug) write(*,*) 'calling HTSIZE...'

c---- set tail CL derivative fraction
      depsda = parg(igdepsda)
      sweeph = parg(igsweeph)
      tanL  = tan(sweep *pi/180.0)
      tanLh = tan(sweeph*pi/180.0)
      ip = ipcruise1
      Mach = para(iaMach,ip)
      beta = sqrt(1.0 - Mach**2)
      dCLhdCL = (beta + 2.0/AR)/(beta + 2.0/ARh)
     &        * sqrt((beta**2 + tanL**2)/(beta**2 + tanLh**2))
     &        * (1.0 - depsda)
      parg(igdCLhdCL) = dCLhdCL

c---- set nacelle CL derivative fraction
      dCLnda  = parg(igdCLnda)
      dCLndCL = dCLnda * (beta + 2.0/AR)
     &        * sqrt(beta**2 + tanL**2)
     &        / (2.0*pi*(1.0 + 0.5*hboxo))
      parg(igdCLndCL) = dCLndCL

cc    if(iHTsize .eq. 1 .or. iterw.le.2) then
      if(iterw.le.2 .and. initwgt .eq. 0) then
c----- size horizontal tail to obtain tail volume coefficient Vh
       lhtail = xhtail - xwing
       Vh = parg(igVh)
       Sh = Vh*S*cma/lhtail
       parg(igSh) = Sh

      else
c----- size horizontal tail and locate wing simultaneously
       call htsize(pari,parg,
     &             para(1,ipdescentn),
     &             para(1,ipcruise1),
     &             para(1,ipcruise1)  )
       xwbox = parg(igxwbox)
       xwing = parg(igxwing)
       lhtail = xhtail - xwing
       Sh = parg(igSh)
       parg(igVh) = Sh*lhtail/(S*cma)

      endif


c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      ip = iprotate
      qstall = 0.5*pare(ierho0,ip)*(pare(ieu0,ip)/1.2)**2
      CDAe = parg(igcdefan) * 0.25*pi*parg(igdfan)**2
      De = qstall*CDAe
      Fe = pare(ieFe,ip)
      Me = (Fe+De)*yeng

      if(iVTsize .eq. 1) then
c----- size vertical tail to obtain tail volume coefficient Vv
       lvtail = xvtail - xwing
       Vv = parg(igVv)
       Sv = Vv*S*b/lvtail
       parg(igSv) = Sv
       parg(igCLveout) = Me/(qstall*Sv*lvtail)

      else
c----- size vertical tail to balance engine-out yaw moment
       lvtail = xvtail - xwing
       CLveout = parg(igCLveout)
       Sv = Me/(qstall*CLveout*lvtail)
       parg(igSv) = Sv
       parg(igVv) = Sv*lvtail/(S*b)

      endif

      if(Ldebug) write(*,*) 'calling TAILPO...'

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- set HT max loading magnitude
      call tailpo(Sh,ARh,lambdah,qne,CLhmax,
     &            bh,coh,poh)
      parg(igbh ) = bh
      parg(igcoh) = coh

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- set VT max loading magnitude, based on single tail + its bottom image
      call tailpo(2.0*Sv/nvtail, 2.0*ARv,lambdav,qne,CLvmax,
     &            bv2,cov,pov)
c---- set actual span of single VT (not including its image)
      bv = 0.5*bv2
      parg(igbv ) = bv
      parg(igcov) = cov

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- size HT primary structure and weight
      if(Ldebug) write(*,*) 'calling HT SURFW...'

      gammah  = lambdah
      gammahs = lambdahs
      ihplan = 0
      Wengh = 0.0
      Ecap = parg(igEcap)
      Eweb = Ecap
      Gcap = Ecap*0.5/(1.0+0.3)
      Gweb = Ecap*0.5/(1.0+0.3)
      call surfw(gee,poh, bh,boh,boh,coh,zsh,
     &           lambdah,lambdahs,gammah,gammahs,
     &           1.0,ihplan,Wengh,
     &           0.0, 0.0, 0.0, 0.0,
     &           sweeph,wboxh,hboxh,hboxh,rhh, fLt,
     &           tauwebh,sigcaph,sigstrut,Ecap,Eweb,Gcap,Gweb,
     &           rhowebh,rhocaph,rhostrut,rhofuel,
     &           Ssh,Msh,tbwebsh,tbcapsh,EIcsh,EInsh,GJsh,
     &           Soh,Moh,tbweboh,tbcapoh,EIcoh,EInoh,GJoh,
     &           dum,dum,dum,
     &         Wscenh,Wsinnh,Wsouth,dxWsinnh,dxWsouth,dyWsinnh,dyWsouth,
     &         Wfcenh,Wfinnh,Wfouth,dxWfinnh,dxWfouth,dyWfinnh,dyWfouth,
     &          Wwebh,  Wcaph,  Wstruth,
     &        dxWwebh,dxWcaph,dxWstruth )

      Whtail   = 2.0*(Wscenh +  Wsinnh +   Wsouth)*(1.0+fhadd)
      dxWhtail = 2.0*(        dxWsinnh + dxWsouth)*(1.0+fhadd)
      parg(igWhtail) = Whtail
      parg(igdxWhtail) = dxWhtail

      parg(igtbwebh) = tbweboh
      parg(igtbcaph) = tbcapoh
      parg(igEIch) = EIcoh
      parg(igEInh) = EInoh
      parg(igGJh)  = GJoh

c---- HT area centroid x-offset from box
      if(Ldebug) write(*,*) 'calling HT SURFDX...'
      call surfdx(bh,boh,boh,lambdah,lambdahs,sweeph,
     &            dxh, macco)
      parg(igxhtail) = xhbox + dxh

c---- set HT piching moment constants
      if(Ldebug) write(*,*) 'calling HT SURFCM...'

      fLoh = 0.
      fLth = fLt
      cmph = 0.
      call surfcm(bh,boh,boh, sweeph, Xaxis,
     &            lambdah,1.0,lambdah,1.0,
     &            ARh,fLoh,fLth, 0.0,0.0,0.0,
     &            CMh0,CMh1)
      do ip = ipstatic, ipdescentn
        para(iaCMh0,ip) = CMh0
        para(iaCMh1,ip) = CMh1
      enddo

c- - - - - - - - - - - - - - - - - - - - - -
c---- size VT (+ its image) primary structure and weight
      if(Ldebug) write(*,*) 'calling VT SURFW...'

      gammav  = lambdav
      gammavs = lambdavs
      ivplan = 0
      Wengv = 0.0
      Ecap = parg(igEcap)
      Eweb = Ecap
      Gcap = Ecap*0.5/(1.0+0.3)
      Gweb = Ecap*0.5/(1.0+0.3)
      call surfw(gee,pov, bv2,bov,bov,cov,zsv,
     &           lambdav,lambdavs,gammav,gammavs,
     &           1.0,ivplan,Wengv,
     &           0.0, 0.0, 0.0, 0.0,
     &           sweepv,wboxv,hboxv,hboxv,rhv, fLt,
     &           tauwebv,sigcapv,sigstrut,Ecap,Eweb,Gcap,Gweb,
     &           rhowebv,rhocapv,rhostrut,rhofuel,
     &           Ssv,Msv,tbwebsv,tbcapsv,EIcsv,EInsv,GJsv,
     &           Sov,Mov,tbwebov,tbcapov,EIcov,EInov,GJov,
     &           dum,dum,dum,
     &         Wscenv,Wsinnv,Wsoutv,dxWsinnv,dxWsoutv,dyWsinnv,dyWsoutv,
     &         Wfcenv,Wfinnv,Wfoutv,dxWfinnv,dxWfoutv,dyWfinnv,dyWfoutv,
     &          Wwebv2,  Wcapv2,  Wstrutv,
     &        dxWwebv2,dxWcapv2,dxWstrutv )

c---- set actual VT tail weights (half of tail+image, times number of tails)
      Wvtail   = (Wscenv +  Wsinnv +   Wsoutv)*(1.0+fvadd) * nvtail
      dxWvtail = (        dxWsinnv + dxWsoutv)*(1.0+fvadd) * nvtail
      parg(igWvtail) = Wvtail
      parg(igdxWvtail) = dxWvtail

      parg(igtbwebv) = tbwebov
      parg(igtbcapv) = tbcapov
      parg(igEIcv) = EIcov
      parg(igEInv) = EInov
      parg(igGJv)  = GJov

c---- VT area centroid x-offset from box
      call surfdx(bv2,bov,bov,lambdav,lambdavs,sweepv,
     &            dxv, macco)
      parg(igxvtail) = xvbox + dxv

c-------------------------------------------------------------------
c---- engine and nacelle sizing and weight section
      WMTO = parg(igWMTO)

      if(Ldebug) write(*,*) 'calling CDSUM...'

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- calculate stuff for start-of-cruise (cruise1) point
      ip = ipcruise1

c---- set pitch trim by adjusting CLh, to get correct wing cl for CDSUM
      Wzero = WMTO - parg(igWfuel)
      Wf = para(iafracW,ip)*WMTO - Wzero
      rfuel = Wf/parg(igWfuel)
      rpay  = 1.0
      xipay = 0.
      itrim = 1
      call balance(pari,parg,para(1,ip),rfuel,rpay,xipay, 
     &             itrim)

c---- calculate overall CD
      icdfun = 1  ! use airfoil database for wing airfoil cdf,cdp
      call cdsum(pari,parg,para(1,ip),pare(1,ip), icdfun,iairf)

c---- size engine for cruise1 point
      DoL  = para(iaCD,ip)/para(iaCL,ip)
      gamV = para(iagamV,ip)
      W = para(iafracW,ip)*WMTO
      BW = W + WbuoyCR
      Fdes = BW * (DoL + gamV)
      pare(ieFe,ip) = Fdes/neng


      icall = 0   ! size engine
      icool = 1   ! use previously-set turbine cooling mass flow

cc    inite1 = 0  
      if(iterw.le.1 .or. initeng.eq.0) then
       inite1 = 0  ! initialize engine state
      else
       inite1 = 1  ! start with current engine state
      endif

      if(Ldebug) write(*,*) 'calling TFCALC...', icall

       call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &             icall,icool,inite1,
     &             ichoke5(ip),ichoke7(ip)) 

cc----- calculate design point as an off-design point, as a check
c       call engwrt(11,cplab(ip),pare(1,ip))
c       icall2 = 2
c       initeng2 = 1
c       call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
c     &             icall2,icool,initeng2,
c     &             ichoke5(ip),ichoke7(ip)) 
c       call engwrt(12,cplab(ip),pare(1,ip))
c       stop

      if(Lengwrt) call engwrt(6,cplab(ip),pare(1,ip))

c---- store engine design-point parameters for all operating points
      parg(igA5) = pare(ieA5,ip) / pare(ieA5fac,ip)
      parg(igA7) = pare(ieA7,ip) / pare(ieA7fac,ip)
      do jp = 1, iptotal
        pare(ieA2 ,jp)   = pare(ieA2 ,ip)  
        pare(ieA25,jp)   = pare(ieA25,ip)  
        pare(ieA5 ,jp)   = parg(igA5) * pare(ieA5fac,jp)
        pare(ieA7 ,jp)   = parg(igA7) * pare(ieA7fac,jp)
                                                   
        pare(ieNbfD ,jp) = pare(ieNbfD ,ip)
        pare(ieNblcD,jp) = pare(ieNblcD,ip) 
        pare(ieNbhcD,jp) = pare(ieNbhcD,ip) 
        pare(ieNbhtD,jp) = pare(ieNbhtD,ip) 
        pare(ieNbltD,jp) = pare(ieNbltD,ip) 
                                                   
        pare(iembfD ,jp) = pare(iembfD ,ip)
        pare(iemblcD,jp) = pare(iemblcD,ip)
        pare(iembhcD,jp) = pare(iembhcD,ip)
        pare(iembhtD,jp) = pare(iembhtD,ip)
        pare(iembltD,jp) = pare(iembltD,ip)
                                                   
        pare(iepifD ,jp) = pare(iepifD ,ip)
        pare(iepilcD,jp) = pare(iepilcD,ip)
        pare(iepihcD,jp) = pare(iepihcD,ip)
        pare(iepihtD,jp) = pare(iepihtD,ip)
        pare(iepiltD,jp) = pare(iepiltD,ip)
      enddo

      dfan   = parg(igdfan)
      dlcomp = parg(igdlcomp)
      dhcomp = parg(igdhcomp)

      Mach = para(iaMach,ip)
      CL = para(iaCL,ip)
      CD = para(iaCD,ip)

c---- bare weight for one engine [Newtons]
      mdotc = pare(iemblcD,ip) * sqrt(Tref/TSL) * (pSL/pref)
      BPR   = pare(ieBPR,ip)
      OPR   = pare(iepilc,ip)*pare(iepihc,ip)

c- - - - - - - - - - - - - - - - - - - - - - - - - - - 
c---- weight of engine and related stuff
c      call tset(time0)
      call tfweight(iengwgt,parg(igGearf),OPR,BPR,mdotc,dfan,rSnace,
     &              dlcomp,neng,feadd,fpylon,
     &              Weng,Wnace,Webare,Snace1)
c      call tadd(time0,t_tfweight)

      parg(igWeng)   = Weng
      parg(igWebare) = Webare
      parg(igWnace ) = Wnace
      parg(igWeng  ) = Weng


c- - - - - - - - - - - - - - - - - - - - - - - - - - - 
c---- set new nacelle area / reference area  fraction fSnace
      Snace = Snace1 * neng
      fSnace = Snace/S
      parg(igfSnace) = fSnace
      lnace = parg(igdfan)*parg(igrSnace)*0.15
      parg(iglnace) = lnace

c-------------------------------------------------------------------
c---- aero calculations and mission-simulation section

      if(Ldebug) write(*,*) 'calling MISSION...'

      ipc1 = 1 ! ipcruise1 aero and engine point is assumed to be calculated
      call mission(pari,parg,parm,para,pare,
     &             iairf,inite1, ipc1,
     &             ichoke5,ichoke7 )

      if(Ldebug) write(*,*) 'MISSION returned...'

c---- this calculated fuel is the design-mission fuel 
      parg(igWfuel) = parm(imWfuel)

c-------------------------------------------------------------------
c---- size cooling mass flow at takeoff rotation condition (at Vstall)
      ip = iprotate

c---- must define CDwing for this point in case there's wing BLI
      cdfw = para(iacdfw,ip) * para(iafexcdw,ip)
      cdpw = para(iacdpw,ip) * para(iafexcdw,ip)
      cosL = cos(parg(igsweep)*pi/180.0)
      para(iaCDwing,ip) = cdfw + cdpw*cosL**3


      icall = 1  ! use fixed engine geometry, specified Tt4
      icool = 2  ! set turbine cooling mass flow
      if(Ldebug) write(*,*) 'calling TFCALC...', icall
      call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &            icall,icool,inite1,
     &            ichoke5(ip),ichoke7(ip)) 

c---- Tmetal was specified... set blade row cooling flow ratios for all points
      do jp = 1, iptotal
        do icrow = 1, ncrowx
          pare(ieepsc1+icrow-1,jp) = pare(ieepsc1+icrow-1,ip)
        enddo
c------ also set first estimate of total cooling mass flow fraction
        pare(iefc,jp) = pare(iefc,ip)
      enddo

      if(Lengwrt) call engwrt(6,cplab(ip),pare(1,ip))

c        Mach = para(iaMach,ip)
c        CL = para(iaCL,ip)
c        CD = para(iaCD,ip)
c        gamV = para(iagamV,ip)
c        write(*,'(/a,f9.5, f10.0, 2f10.5, 2f9.4, f10.5, f9.4)')
c     &   ' climb:  fW W F/W D/L M CL CD gam',
c     &     para(iafracW,ip), W, F/W, DoL, Mach, CL, CD, gamV*180.0/pi

c-----------------------------------------------------------------
c---- recalculate max weight with latest weight info
      ip = ipcruise1
      call Wupdate(parg,rlx,fsum)
      if(fsum .ge. 1.0) go to 110

      parm(imWTO)   = parg(igWMTO)
      parm(imWfuel) = parg(igWfuel)

c-----------------------------------------------------------------
c---- set previous-iteration weights for next iteration
      WMTO3 = WMTO2
      WMTO2 = WMTO1
      WMTO1 = parg(igWMTO)

 100  continue
c=======================================================================
 110  continue
      write(*,*) 
     & 'WSIZE: Weight iteration not converged.  dWrel =',errw, fsum
cc      return
c
c---- jump to here if converged successfully
 150  continue

c-----------------------------------------------------------------
c---- normal takeoff and balanced-field takeoff calculations

c      write(*,*) 'static...'
c---- set static thrust for takeoff routine
      ip = ipstatic
      icall = 1
      icool = 1
      call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &            icall,icool,inite1,
     &            ichoke5(ip),ichoke7(ip)) 

c---- set rotation thrust for takeoff routine
c-     (already available from cooling calculations)
      ip = iprotate
      icall = 1
      icool = 1
      call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &            icall,icool,inite1,
     &            ichoke5(ip),ichoke7(ip)) 

c---- calculate takeoff and balanced-field lengths
      call takeoff(pari,parg,parm,para,pare,
     &             inite1,
     &             ichoke5,ichoke7)

cc---- calculate noise
c      call noise(pari,parg,parm(1,km),para(1,1,km),pare(1,1,km),
c     &               inite1,iairf,ichoke5(1,km),ichoke7(1,km))
c      call htsize(pari,parg,
c     &            para(1,ipdescentn),
c     &            para(1,ipcruise1),
c     &            para(1,ipcruise1)  )

      if(Ldebug) write(*,*) 'calling CGLPAY...'

c---- calculate CG limits from worst-case payload fractions and packings
      call cglpay(parg,
     &            rfuel0, rpay0, xCG0,
     &            rfuel1, rpay1, xCG1 )
      parg(igxCGfwd) = xCG0
      parg(igxCGaft) = xCG1
      parg(igrpayfwd) = rpay0
      parg(igrpayaft) = rpay1

      if(Ldebug) write(*,*) 'calling BALANCE...'

c---- set neutral point at cruise
      ip = ipcruise1
      Wzero = WMTO - parg(igWfuel)
      Wf = para(iafracW,ip)*WMTO - Wzero
      rfuel = Wf/parg(igWfuel)
      rpay  = 1.0
      xipay = 0.0
      itrim = 0
      call balance(pari,parg,para(1,ip), rfuel,rpay,xipay, 
     &             itrim)
      parg(igxNP) = para(iaxNP,ip)

c-----------------------------------------------------------------

c---- INSTRUMENTATION: dump the complete output state on exit.
      open(87,file='wsize_out.txt',status='unknown')
      write(87,'(e26.18)') (parg(k), k=1,igtotal)
      write(87,'(e26.18)') (parm(k), k=1,imtotal)
      do k = 1, iatotal
        write(87,'(e26.18)') (para(k,ipx), ipx=1,iptotal)
      enddo
      do k = 1, ietotal
        write(87,'(e26.18)') (pare(k,ipx), ipx=1,iptotal)
      enddo
      write(87,'(i6)') iterw
      if(Lconv) then
        write(87,'(i6)') 1
      else
        write(87,'(i6)') 0
      endif
      write(87,'(e26.18)') errw
      close(87)

      if(Ldebug) write(*,*) 'exiting WSIZE...'

      return
      end ! wsize




      subroutine Wupdate(parg,rlx,fsum)
      implicit real (a-h,l-z)
      include 'index.inc'
      real parg(igtotal)

c     call Wupdate1(parg,rlx,fsum)
c     return

      WMTO = parg(igWMTO)

      fwing  = parg(igWwing )/WMTO
      fstrut = parg(igWstrut)/WMTO
      fhtail = parg(igWhtail)/WMTO
      fvtail = parg(igWvtail)/WMTO
      feng   = parg(igWeng  )/WMTO
      ffuel  = parg(igWfuel )/WMTO
      fhpesys = parg(igfhpesys)
      flgnose = parg(igflgnose)
      flgmain = parg(igflgmain)

      Wpay    = parg(igWpay)
      Wfuse   = parg(igWfuse)


c      Wfuse = Wfix + Wapu + Wpadd + Wseat
c     &      + Wshell + Wcone + Wwindow + Winsul + Wfloor
c     &      + Whbend + Wvbend

c      WMTO = Wpay + Wfuse
c    &      + Wwing + Wstrut + Whtail + Wvtail
c    &      + Weng + Wfuel
c    &      + Whpesys + Wlgnose + Wlgmain


c---- if fsum > 1, then weight exploded... go exit
      fsum = fwing + fstrut
     &     + fhtail + fvtail
     &     + feng + ffuel
     &     + fhpesys + flgnose + flgmain

      if(fsum .ge. 1.0) return

      WMTO = rlx*(Wpay + Wfuse)/(1.0-fsum) + (1.0-rlx)*WMTO

      parg(igWMTO )   = WMTO
      parg(igWwing)   = WMTO*fwing
      parg(igWstrut)  = WMTO*fstrut
      parg(igWhtail)  = WMTO*fhtail
      parg(igWvtail)  = WMTO*fvtail
      parg(igWeng )   = WMTO*feng
      parg(igWfuel)   = WMTO*ffuel

      return
      end ! Wupdate




      subroutine Wupdate1(parg,rlx,fsum)
      implicit real (a-h,l-z)
      include 'index.inc'
      real parg(igtotal)

      WMTO = parg(igWMTO)

c      Wfuse = Wfix + Wapu + Wpadd + Wseat
c     &      + Wshell + Wcone + Wwindow + Winsul + Wfloor
c     &      + Whbend + Wvbend

c      WMTO = Wpay + Wfuse
c    &      + Wwing + Wstrut + Whtail + Wvtail
c    &      + Weng + Wfuel
c    &      + Whpesys + Wlgnose + Wlgmain


      fwing  = parg(igWwing )/WMTO
      fstrut = parg(igWstrut)/WMTO
      fhbend = parg(igWhbend)/WMTO
      fvbend = parg(igWvbend)/WMTO
      fcone  = parg(igWcone )/WMTO
      fhtail = parg(igWhtail)/WMTO
      fvtail = parg(igWvtail)/WMTO
      feng   = parg(igWeng  )/WMTO
      ffuel  = parg(igWfuel )/WMTO

      ftotadd = parg(igfhpesys)
     &        + parg(igflgnose)
     &        + parg(igflgmain)


      Wfix    = parg(igWfix)
      Wpay    = parg(igWpay)
      Wapu    = parg(igWpay)*parg(igfapu)
      Wpadd   = parg(igWpay)*parg(igfpadd)
      Wseat   = parg(igWpay)*parg(igfseat)
      Wshell  = parg(igWshell)
      Wwindow = parg(igWwindow)
      Winsul  = parg(igWinsul)
      Wfloor  = parg(igWfloor)

c      Wfuse   = parg(igWfuse)
c      Wfuse = Wfix + Wapu + Wpadd + Wseat
c     &      + Wshell + Wcone + Wwindow + Winsul + Wfloor
c     &      + Whbend + Wvbend


c---- if fsum > 1, then weight exploded... go exit
      fsum = fwing + fstrut
     &     + fhbend + fvbend + fcone
     &     + fhtail + fvtail
     &     + feng + ffuel
     &     + ftotadd

      if(fsum .ge. 1.0) return

      WMTO = rlx*(Wfix + Wpay + Wpadd + Wapu + Wshell
     &           + Wwindow + Winsul + Wfloor + Wseat) / (1.0-fsum)
     &     + (1.0-rlx)*WMTO

      Wwing   = WMTO*fwing
      Wstrut  = WMTO*fstrut
      Whbend  = WMTO*fhbend
      Wvbend  = WMTO*fvbend
      Wcone   = WMTO*fcone
      Whtail  = WMTO*fhtail
      Wvtail  = WMTO*fvtail
      Weng    = WMTO*feng
      Wfuel   = WMTO*ffuel 

      parg(igWMTO )   = WMTO
      parg(igWwing)   = WMTO*fwing
      parg(igWstrut)  = WMTO*fstrut
      parg(igWhbend)  = WMTO*fhbend
      parg(igWvbend)  = WMTO*fvbend
      parg(igWcone)   = WMTO*fcone
      parg(igWhtail)  = WMTO*fhtail
      parg(igWvtail)  = WMTO*fvtail
      parg(igWeng )   = WMTO*feng
      parg(igWfuel)   = WMTO*ffuel

      Wfuse = Wfix
     &      + Wapu
     &      + Wpadd
     &      + Wseat
     &      + Wshell
     &      + parg(igWcone)
     &      + Wwindow
     &      + Winsul
     &      + Wfloor
     &      + parg(igWhbend)
     &      + parg(igWvbend)
      parg(igWfuse) = Wfuse

      return
      end ! Wupdate1



      subroutine Wupdate0(parg,rlx,fsum)
      implicit real (a-h,l-z)
      include 'index.inc'
      real parg(igtotal)

c---- total max takeoff weight
c      WMTO = Wpay + Wfuse
c    &      + Wwing + Wstrut + Whtail + Wvtail
c    &      + Weng + Wfuel
c    &      + Whpesys + Wlgnose + Wlgmain

      WMTO = parg(igWMTO)

!      feng  = parg(igWeng) /WMTO
!      ffuel = parg(igWfuel)/WMTO

      ftotadd = parg(igfhpesys)
     &        + parg(igflgnose)
     &        + parg(igflgmain)
!     &        + feng
!     &        + ffuel

      fsum = 0.0

      Wsum = parg(igWpay)
     &     + parg(igWfuse)
     &     + parg(igWwing )
     &     + parg(igWstrut)
     &     + parg(igWhtail)
     &     + parg(igWvtail)
     &     + parg(igWeng  )
     &     + parg(igWfuel )


cc    WMTO = Wsum + WMTO*ftotadd
cc    WMTO = Wsum/(1.0-ftoadd)

      WMTO = rlx*Wsum/(1.0-ftotadd) + (1.0-rlx)*WMTO

      parg(igWMTO) = WMTO

!      parg(igWeng ) = WMTO*feng
!      parg(igWfuel) = WMTO*ffuel

      return
      end ! Wupdate0



      real function cfturb(re)
c-------------------------------------------------------------
c     Returns total Cf for turbulent flat plate, versus Re_l
c-------------------------------------------------------------
ccc   cfturb = 0.427/(log10(re) - 0.407)**2.64   ! original Hoerner
ccc   cfturb = 0.310/(log10(re) - 0.407)**2.47   ! modified (weaker dependence on Re)
c
      cfturb = 0.523/(log(0.06*re))**2           ! White
c
      return
      end ! cfturb


      subroutine pralt(p0spec,gee,
     &                 altkm, T0,p0,rho0,a0,mu0)
c--------------------------------------------------------------
c     Returns standard altitude and atmospheric properties
c     for specified pressure p0spec
c--------------------------------------------------------------
      implicit real(a-h,l-z)

      do iterh = 1, 15
        call atmos(altkm, T0,p0,rho0,a0,mu0)
        delp = p0 - p0spec
        dpdh = -rho0*gee
        dh = -delp/dpdh
cc      write(*,'(1x,i3,5g13.5)') iterh, altkm, delp, dh
        if(abs(dh) .lt. 0.001) return
        altkm = altkm + dh/1000.0
      enddo
      write(*,*) 'pralt: pressure-altitude calc failed, dp =', delp
      return
      end ! pralt



      real function muair(T)
      implicit real(a-h,l-z)
      data muref, Tref, Ts / 1.78e-5 , 288.0 , 110.0 /

      muair = muref * sqrt(T/Tref)**3 * (Tref+Ts)/(T+Ts)

      return
      end ! muair
