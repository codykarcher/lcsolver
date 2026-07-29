
      subroutine woper(pari,parg,parm,para,pare,  parad,pared,
     &                 iterfmax,initeng,iairf,
     &                 ichoke5,ichoke7,
     &                 Litprint,Lconv)
c=====================================================================
c     Calculates performance for off-design mission
c     
c     Input:
c      pari(.)   flags
c      parg(.)   geometry parameters
c      parad(.)  para(.) paramerers for design mission 
c      pared(.)  pare(.) paramerers for design mission
c      iterfmax  max number of fuel weight iterations
c      initeng    0 = initialize engine states for engine calcs
c                 1 = use existing states to start engine calcs
c      iairf     index of airfoil database to use
c      Litprint  T = print weight-iteration history
c
c     Input/Output:
c      parm(.)   sizing   parameters
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
     &     pare(ietotal,iptotal),
     &     parad(iatotal,iptotal),
     &     pared(ietotal,iptotal)
      integer iterfmax,ifuel,initeng
      integer ichoke5(iptotal),
     &        ichoke7(iptotal)
      logical Litprint,Lconv
c
      include 'airf.inc'
      include 'constants.inc'

      logical Lengwrt, Lprint, Ldebug
      real KAfTE

      integer lu

      common /com_ip/ ip

c---- convergence tolerance, fractional weight change between iterations
      data tolerw / 1.0e-9 /
c      data tolerw / 1.0e-10 /
c      data tolerw / 1.0e-14 /

      Lengwrt = .false.   ! don't write engine parameters below
c      Lengwrt = .true.    ! write engine parameters below (for debugging)

      Ldebug = .false.
c      Ldebug = .true.

      if(Ldebug) write(*,*) 'entering WOPER...'

c---- INSTRUMENTATION: dump the complete input state on entry.
c---- Not part of TASOPT -- see fortran_ref/woper_instrumented.f.
      open(84,file='woper_in.txt',status='unknown')
      write(84,'(i6)') (pari(k), k=1,iitotal)
      write(84,'(e26.18)') (parg(k), k=1,igtotal)
      write(84,'(e26.18)') (parm(k), k=1,imtotal)
      do k = 1, iatotal
        write(84,'(e26.18)') (para(k,ipx), ipx=1,iptotal)
      enddo
      do k = 1, ietotal
        write(84,'(e26.18)') (pare(k,ipx), ipx=1,iptotal)
      enddo
      do k = 1, iatotal
        write(84,'(e26.18)') (parad(k,ipx), ipx=1,iptotal)
      enddo
      do k = 1, ietotal
        write(84,'(e26.18)') (pared(k,ipx), ipx=1,iptotal)
      enddo
      write(84,'(i6)') iterfmax, initeng, iairf
      close(84)


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


      do ip = 1, iptotal
        do ia = 1, iatotal
          para(ia,ip) = parad(ia,ip)
        enddo

c------ mission-varying excrescence factors disabled in this version
c-      ( also commented out in getparm.f )
c        para(iafexcdw,ip) = parm(imfexcdw)
c        para(iafexcdt,ip) = parm(imfexcdt)
c        para(iafexcdf,ip) = parm(imfexcdf)

        do ie = 1, ietotal
          pare(ie,ip) = pared(ie,ip)
        enddo
      enddo

      ip = ipcruise1
      call fusebl(pari,parg,para(1,ip))

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

C===================================================================
c---- max range and this mission range
      Rangemax = parg(igRange)
      Rangetot = parm(imRange)

c---- max TO weight
      WMTO = parg(igWMTO)

c---- zero-fuel weight for this mission
      Wzero = WMTO
     &      - parg(igWfuel)
     &      - parg(igWpay)
     &      + parm(imWpay)

C===================================================================
c---- initial fuel and gross takeoff weight estimates from Breguet, R ~ ln(1+f)
      gmax = log(1.0 + parg(igWfuel)/Wzero)
      gmaxp = gmax * Rangetot/Rangemax
      Wfuel = (exp(gmaxp) - 1.0) * Wzero
      WTO = Wzero + Wfuel

      parm(imWfuel) = Wfuel
      parm(imWTO)   = WTO

c---- scale initial weight fractions by takeoff and descent weight ratios
      rTO = WTO/WMTO
      rDE = Wzero/(WMTO-parg(igWfuel))

      if(Ldebug) write(*,*) 'rTO rDE', rTO, rDE

      para(iafracW,ipstatic ) = parad(iafracW,ipstatic )*rTO
      para(iafracW,iprotate ) = parad(iafracW,iprotate )*rTO
      para(iafracW,iptakeoff) = parad(iafracW,iptakeoff)*rTO
      para(iafracW,ipcutback) = parad(iafracW,ipcutback)*rTO
      do ip = ipclimb1, ipclimbn
        para(iafracW,ip) = parad(iafracW,ip) * rTO
      enddo
      do ip = ipcruise1, ipcruisen
        frac = float(ip       -ipcruise1)
     &       / float(ipcruisen-ipcruise1)
        rCR = rTO*(1.0-frac) + rDE*frac
        para(iafracW,ip) = parad(iafracW,ip) * rCR
      enddo
      do ip = ipdescent1, ipdescentn
        para(iafracW,ip) = parad(iafracW,ip) * rDE
      enddo

      do ip = 1, iptotal
        para(iagamV,ip) = parad(iagamV,ip)
      enddo


      if(Ldebug) write(*,*) 'TO speed calc...'

c---- estimate takeoff speed and set V,Re over climb and descent
c-    (needed to start trajectory integration)
      ip = iptakeoff
      VTO = pared(ieu0,ip) * sqrt(pared(ierho0,ip)/pare(ierho0,ip))
      ReTO = VTO*pare(ierho0,ip)/pare(iemu0,ip)

      ip = ipcruise1
      VCR = pared(ieu0,ip)
      ReCR = parad(iaReunit,ip)

      do ip = iprotate, ipclimb1
        pare(ieu0,ip) = VTO
        para(iaReunit,ip) = ReTO
      enddo
      do ip = ipclimb1+1, ipclimbn
        frac = float(ip-ipclimb1) / float(ipclimbn-ipclimb1)
        V  =  VTO*(1.0-frac) +  VCR*frac
        Re = ReTO*(1.0-frac) + ReCR*frac
        pare(ieu0,ip) = V
        para(iaReunit,ip) = Re
      enddo
      do ip = ipdescent1, ipdescentn
        frac = float(ip-ipdescent1) / float(ipdescentn-ipdescent1)
        V  =  VTO*frac +  VCR*(1.0-frac)
        Re = ReTO*frac + ReCR*(1.0-frac)
        pare(ieu0,ip) = V
        para(iaReunit,ip) = Re
      enddo

      if(initeng.eq.0) then
c----- use design case as initial guess for engine state
       do ip = 1, iptotal
         do ie = 1, ietotal
           pare(ie,ip) = pared(ie,ip)
         enddo
       enddo
      endif
    
      do ip = ipstatic, ipdescentn
        para(iaCfnace,ip) = parad(iaCfnace,ip)
      enddo

c--------------------------------------------------------------------------
c---- set wing pitching moment constants
      b  = parg(igb)
      bs = parg(igbs)
      bo = parg(igbo)
      sweep = parg(igsweep)
      Xaxis = parg(igXaxis)
      lambdas = parg(iglambdas)
      lambdat = parg(iglambdat)
      AR = parg(igAR)
      fLo = parg(igfLo)
      fLt = parg(igfLt)

      ip = iptakeoff
      cmpo = para(iacmpo,ip)
      cmps = para(iacmps,ip)
      cmpt = para(iacmpt,ip)
      gammat = parg(iglambdat)*para(iarclt,ip)
      gammas = parg(iglambdas)*para(iarcls,ip)
      if(Ldebug) write(*,*) 'calling TO SURFCM...'

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
      if(Ldebug) write(*,*) 'calling CR SURFCM...'
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
      if(Ldebug) write(*,*) 'calling DE SURFCM...'
      call surfcm(b,bs,bo, sweep, Xaxis,
     &            lambdat,lambdas,gammat,gammas,
     &            AR,fLo,fLt,cmpo,cmps,cmpt,
     &            CMw0,CMw1)
      do ip = ipdescentn, ipdescentn
        para(iaCMw0,ip) = CMw0
        para(iaCMw1,ip) = CMw1
      enddo

c---- tail pitching moment constants
      bh      = parg(igbh)
      boh     = parg(igboh)
      sweeph  = parg(igsweeph)
      lambdah = parg(iglambdah)
      ARh     = parg(igARh)
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

c---- initial guesses for nacelle wetted-area Cf
      Cfnace = 0.003
      do ip = 1, iptotal
        para(iaCfnace,ip) = Cfnace 
      enddo

c---- initialize previous-iteration weights (none yet)
      WTO1 = 0.0  ! 2nd-previous-iteration weight, for convergence criterion
      WTO2 = 0.0  ! 1st-previous-iteration weight, for convergence criterion

c---- no convergence yet
      Lconv = .false.

c---- set these to zero for first-iteration info printout
      do ip = 1, iptotal
        ichoke5(ip) = 0
        ichoke7(ip) = 0
      enddo

c==============================================================================
c---- start weight-iteration loop
      do 100 iterw = 1, iterfmax

      rlx = 1.0
      if(iterw .gt. iterfmax-5) then
        rlx = 0.5
      endif

c---- scale initial cruise altitude by p ~ W 
      ip = ipcruise1
      altkm = para(iaalt,ip)/1000.0
      p0new = pared(iep0,ip) * para(iafracW,ip)/parad(iafracW,ip)
      call pralt(p0new,gee, altkm, T0,p0,rho0,a0,mu0)

      para(iaalt,ip) = altkm*1000.0
      Mach = para(iaMach,ip)
      para(iaReunit,ip) = Mach*a0*rho0/mu0


c======================================================================
c---- generate high-speed polars
c      do icl = 4, 9
c        para(iaCL,ip) = float(icl) / 10.0
c
c        ilu = icl + 16
c        write(ilu,*) 
c     &'#      CL      CD       CDi      cd      CDwing    CL/CD      Ma'
c        do imach = 500, 850, 5
c          para(iaMach,ip) = float(imach) / 1000.0
c
c          rfuel = 1.0
c          rpay  = 1.0
c          xipay = 0.
c          itrim = 1
c          call balance(pari,parg,para(1,ip),rfuel,rpay,xipay, 
c     &                 itrim)
c
cc-------- calculate overall CD
c          icdfun = 1  ! use airfoil database for wing airfoil cdf,cdp
c          call cdsum(pari,parg,para(1,ip),pare(1,ip), icdfun,iairf)
c          LoD  = para(iaCL,ip)/para(iaCD,ip)
c          cdprof = para(iacdfw,ip) + para(iacdpw,ip)
c
c          write(ilu,'(1x,f10.4,4f10.6,f10.4,f10.4)')
c     &      para(iaCL,ip),
c     &      para(iaCD,ip),
c     &      para(iaCDi,ip),
c     &      cdprof,
c     &      para(iaCDwing,ip),
c     &      LoD,
c     &      para(iaMach,ip)
c        enddo
c      enddo
c
c      stop
cc======================================================================


      if(Ldebug) write(*,*)'calling MISSION...',parm(imWTO)/parg(igWMTO)

      ipc1 = 0  ! ipcruise1 aero and engine point needs to be calculated
      call mission(pari,parg,parm,para,pare,
     &             iairf,initeng, ipc1,
     &             ichoke5,ichoke7 )

      if(Ldebug) write(*,*) '...exited from MISSION'

c---- must define CDwing for this point in case there's wing BLI
      ip = iprotate
      cdfw = para(iacdfw,ip) * para(iafexcdw,ip)
      cdpw = para(iacdpw,ip) * para(iafexcdw,ip)
      cosL = cos(parg(igsweep)*pi/180.0)
      para(iaCDwing,ip) = cdfw + cdpw*cosL**3

c-------------------------------------------------------------------------
c---- convergence tests

c---- set max error from last 2 iterations 
c-    (prevents false convergence from "lucky" near-zero single change)
      WTO = parm(imWTO)
      errw1 = (WTO-WTO1)/WTO
      errw2 = (WTO-WTO2)/WTO
      errw = max( abs(errw1) , abs(errw2) )

      if(Litprint) then
        if(iterw.eq.1) then
          write(*,'(/1x,a,a,a,a,a)')
     & ' iterw     errW     ',
     & '      WTO        Wfuel   ',
     & '    h_CR1     h_CR2    ',
     & '   gam_BOC   gam_TOC   '
        endif

 4100   format(1x, i5, f14.10, 2f13.4, 2f11.2, 2f11.5)
        write(*,4100)
     &     iterw, errw2,
     &     parm(imWTO)*lb_N,
     &     parm(imWfuel)*lb_N,
     &     para(iaalt,ipcruise1)*ft_m,
     &     para(iaalt,ipcruisen)*ft_m,
     &     para(iagamV,ipclimb1)*180.0/pi,
     &     para(iagamV,ipclimbn)*180.0/pi
      endif

      if(errw .lt. tolerw) then
        Lconv = .true.
        go to 150
      endif

c-----------------------------------------------------------------
c---- set previous-iteration weights for next iteration
      WTO1 = WTO2  
      WTO2 = parm(imWTO)

 100  continue
c=======================================================================
 110  continue
      write(*,*) 'WOPER: Weight iteration not converged.  dWrel =',errw
cc    return
c
c---- jump to here if converged successfully
 150  continue

c-----------------------------------------------------------------
c---- normal takeoff and balanced-field takeoff calculations

c---- set static thrust for takeoff routine
      ip = ipstatic
      icall = 1
      icool = 1
      if(Ldebug) write(*,*) 'calling TFCALC static...',pare(ieTt4,ip)
      call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &            icall,icool,initeng,
     &            ichoke5(ip),ichoke7(ip)) 

c---- set rotation thrust for takeoff routine
      ip = iprotate
      icall = 1
      icool = 1
      if(Ldebug) write(*,*) 'calling TFCALC rotate...', pare(ieTt4,ip)
      call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &            icall,icool,initeng,
     &            ichoke5(ip),ichoke7(ip)) 

c---- calculate takeoff and balanced-field lengths
      if(Ldebug) write(*,*) 'calling TAKEOFF...'
      call takeoff(pari,parg,parm,para,pare,
     &             initeng,
     &             ichoke5,ichoke7)

cc---- calculate noise
c          call noise(pari,parg,parm(1,km),para(1,1,km),pare(1,1,km),
c     &               initeng,iairf,ichoke5(1,km),ichoke7(1,km))

c-----------------------------------------------------------------

c---- INSTRUMENTATION: dump the complete output state on exit.
      open(85,file='woper_out.txt',status='unknown')
      write(85,'(e26.18)') (parm(k), k=1,imtotal)
      do k = 1, iatotal
        write(85,'(e26.18)') (para(k,ipx), ipx=1,iptotal)
      enddo
      do k = 1, ietotal
        write(85,'(e26.18)') (pare(k,ipx), ipx=1,iptotal)
      enddo
      write(85,'(i6)') iterw
      if(Lconv) then
        write(85,'(i6)') 1
      else
        write(85,'(i6)') 0
      endif
      write(85,'(e26.18)') errw
      close(85)

      if(Ldebug) write(*,*) 'exiting WOPER...'

      return
      end ! woper

