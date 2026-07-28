
      subroutine mission(pari,parg,parm,para,pare,
     &                   iairf,initeng, ipc1,
     &                   ichoke5,ichoke7 )
c=====================================================================
c     Runs aircraft through mission, calculating fuel burn
c     and other mission variables.
c     
c     Input:
c      pari(.)   integer flags
c      parg(.)   geometry parameters
c      parm(.)   mission parameters
c      iairf     index of airfoil database to use
c      initeng    0 = engine state will be initialized for all points
c                 1 = engine state is assumed to be initialized
c      ipc1       0 = ipcruise1 aero and engine point needs to be calculated
c                 1 = ipcruise1 aero and engine point assumed calculated
c
c     Input/Output:
c      para(.p)  aero     parameters for points p=1..iptotal
c      pare(.p)  engine   parameters for points p=1..iptotal
c
c     Output:
c      ichoke5(p)  0 = core nozzle unchoked, 1 = core nozzle choked
c      ichoke7(p)  0 = fan  nozzle unchoked, 1 = fan  nozzle choked
c
c
c     NOTE: 
c      This routine assumes that estimates of the climb-leg flight path 
c      gamma angles are passed in via para(iagamV,ipclimb1:ipclimbn).
c      These appear as cos(gamma) factors in the climb equations,
c      and can be passed in as zero with only a minor error.
c      They are updated and returned in the same para(iagamV,ip) array.
c
c=====================================================================
      implicit real (a-h,l-z)
c
      include 'index.inc'
      integer pari(iitotal)
      real parg(igtotal), 
     &     parm(imtotal),
     &     para(iatotal,iptotal),
     &     pare(ietotal,iptotal)
      integer ifuel,initeng
      integer k, ipx
      integer ichoke5(iptotal),
     &        ichoke7(iptotal)
c
      include 'airf.inc'
      include 'constants.inc'

      integer ncrow, icrow

c---- local mission-point arrays, for integration of trajectory equations
      real FoW(iptotal),
     &     FFC(iptotal),
     &     Vgi(iptotal)

      logical Lengwrt, Ldebug

      common /com_ip/ ip

      data itergmax / 10 /
      data gamVtol / 1.0e-12 /

      Lengwrt = .false.   ! don't write engine parameters below
c      Lengwrt = .true.    ! write engine parameters below (for debugging)

      Ldebug = .false.
c      Ldebug = .true.

      if(Ldebug) write(*,*) 'entering MISSION...'

c---- unpack flags
      iengloc = pari(iiengloc)
      ifclose = pari(iifclose)
      ifuel   = pari(iifuel  )

c---- dump the complete input state for the port's reference
      open(86,file='mission_in.txt',status='unknown')
      write(86,'(i6)') (pari(k), k=1,iitotal)
      write(86,'(e26.18)') (parg(k), k=1,igtotal)
      write(86,'(e26.18)') (parm(k), k=1,imtotal)
      do k = 1, iatotal
        write(86,'(e26.18)') (para(k,ipx), ipx=1,iptotal)
      enddo
      do k = 1, ietotal
        write(86,'(e26.18)') (pare(k,ipx), ipx=1,iptotal)
      enddo
      write(86,'(2i6)') iairf, initeng
      write(86,'(i6)') ipc1
      close(86)
c---- mission range
      Rangetot = parm(imRange)
      para(iaRange,ipdescentn) = Rangetot

c---- max TO weight
      WMTO = parg(igWMTO)

c---- payload fraction for this mission
      rpay = parm(imWpay)/parg(igWpay)
      xipay = 0.

c---- zero-fuel weight for this mission
      Wzero = WMTO
     &      - parg(igWfuel)
     &      - parg(igWpay)
     &      + parm(imWpay)

c---- mission TO weight
      WTO = parm(imWTO)

c---- mission TO fuel weight
      WfTO = WTO - Wzero


c------------------------------------------------------------------
c---- set known operating conditions

c---- takeoff altitude conditions
      ip = ipstatic
      altkm = para(iaalt,ip)/1000.0
      call atmos(altkm, T_std,p_std,rho_std,a_std,mu_std)
      T0 = parm(imT0TO)
      p0   = p_std
      rho0 = rho_std*(T_std/T0)
      a0   = a_std*sqrt(T0/T_std)
      mu0  = mu_std*(T0/T_std) ** 0.8

      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      pare(ieM0  ,ip) = 0.0
      pare(ieu0  ,ip) = 0.0

      para(iaMach,ip) = 0.0
      para(iaCL  ,ip) = 0.0
      para(iaReunit,ip) = 0.0
      para(iaWbuoy,ip) = 0.

      ip = iprotate
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      para(iaWbuoy,ip) = 0.
 
      ip = iptakeoff
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      para(iaWbuoy,ip) = 0.
 
      ip = ipcutback
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      para(iaWbuoy,ip) = 0.
 
      ip = ipclimb1
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      para(iaWbuoy,ip) = 0.

c---- start-of-cruise altitude conditions
      ip = ipcruise1
      Mach = para(iaMach,ip)
      altkm = para(iaalt,ip)/1000.0
      
      call atmos(altkm, T0,p0,rho0,a0,mu0)
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      pare(ieM0  ,ip)   = Mach
      pare(ieu0  ,ip)   = Mach*a0
      para(iaReunit,ip) = Mach*a0 * rho0/mu0

c---- end-of-descent altitude conditions
      ip = ipdescentn
      altkm = para(iaalt,ip)/1000.0
      call atmos(altkm, T_std,p_std,rho_std,a_std,mu_std)
      T0 = parm(imT0TO)
      p0   = p_std
      rho0 = rho_std*(T_std/T0)
      a0   = a_std*sqrt(T0/T_std)
      mu0  = mu_std*(T0/T_std) ** 0.8
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      para(iaWbuoy,ip) = 0.

c---- interpolate CL over climb points, 
c-     between specified ipclimb1+1, ipclimbn values
      CLa = para(iaCL,ipclimb1+1)
      CLb = para(iaCL,ipcruise1 )

      do ip = ipclimb1+1, ipclimbn
        frac = float(ip      -(ipclimb1+1))
     &       / float(ipclimbn-(ipclimb1+1))
        para(iaCL,ip) = CLa*(1.0-frac**2)
     &                + CLb*     frac**2
      enddo

c---- interpolate CL over descent points, 
c-     between specified ipdescent1, ipdescentn-1 values
c      CLd = para(iaCL,ipcruisen)
c      CLe = para(iaCL,ipcruisen)
      CLd = para(iaCL,ipcruisen) * 0.96
      CLe = para(iaCL,ipcruisen) * 0.50

      do ip = ipdescent1, ipdescentn-1
        frac = float( ip           -ipdescent1)
     &       / float((ipdescentn-1)-ipdescent1)
        fb = 1.0-frac
        para(iaCL,ip) = CLd*     fb**2
     &                + CLe*(1.0-fb**2)
      enddo

c---- interpolate altitudes over climb
      altb = para(iaalt,iptakeoff)
      altc = para(iaalt,ipcruise1)
      do ip = ipclimb1+1, ipclimbn
        frac = float(ip-ipclimb1) / float(ipclimbn-ipclimb1)
        para(iaalt,ip) = altb*(1.0-frac) + altc*frac
      enddo

c---- estimate takeoff speed and set V,Re over climb and descent
      cosL = cos(parg(igsweep)*pi/180.0)
      CLTO = para(iaclpmax,iptakeoff)*cosL**2
      VTO = pare(ieu0,ipcruise1)
     &    * sqrt(pare(ierho0,ipcruise1) / pare(ierho0,iptakeoff))
     &    * sqrt(para(iaCL  ,ipcruise1) / CLTO )
      ReTO = VTO*pare(ierho0,iptakeoff)/pare(iemu0 ,iptakeoff)
      do ip = iprotate, ipclimb1
        pare(ieu0,ip) = VTO
        para(iaReunit,ip) = ReTO
      enddo
      do ip = ipclimb1+1, ipclimbn
        frac = float(ip-ipclimb1) / float(ipclimbn-ipclimb1)
        V  =  VTO*(1.0-frac) + pare(ieu0,ipcruise1)*frac
        Re = ReTO*(1.0-frac) + para(iaReunit,ipcruise1)*frac
        pare(ieu0,ip) = V
        para(iaReunit,ip) = Re
      enddo
      do ip = ipdescent1, ipdescentn
        frac = float(ip-ipdescent1) / float(ipdescentn-ipdescent1)
        V  =  VTO*frac + pare(ieu0,ipcruisen)*(1.0-frac)
        Re = ReTO*frac + para(iaReunit,ipcruisen)*(1.0-frac)
        pare(ieu0,ip) = V
        para(iaReunit,ip) = Re
      enddo


      if(Ldebug) write(*,*) 'takeoff setup...'

c-----------------------------------------------------------------
c---- takeoff CLmax via section clmax and sweep correction
      ip = iprotate
      clpmax = para(iaclpmax,ip)
      sweep = parg(igsweep)
      cosL = cos(sweep*pi/180.0)
      CLmax = clpmax * cosL**2
cc    CLmax = clpmax * cosL

c---- Vs stall speed (takeoff condition)
      rho0 = pare(ierho0,ip)
      a0   = pare(iea0  ,ip)
      S    = parg(igS)
      Vstall = sqrt(2.0*WTO/(rho0*S*CLmax))
      Mstall = Vstall/a0
      pare(ieu0,ip) = Vstall
      pare(ieM0,ip) = Mstall
      para(iaMach,ip) = Mstall
      para(iaReunit,ip) = Vstall*pare(ierho0,ip)
     &                         / pare(iemu0 ,ip)

c---- V2 speed per FAR-25  (takeoff,cutback,climb1 condition)
      V2 = Vstall*1.2
      M2 = Mstall*1.2
      CL2 = CLmax/1.2**2

      ip = iptakeoff
      pare(ieu0    ,ip) = V2
      pare(ieM0    ,ip) = M2
      para(iaMach  ,ip) = M2
      para(iaReunit,ip) = V2*pare(ierho0,ip) / pare(iemu0 ,ip)
      para(iaCL,ip) = CL2

      if(Ldebug) write(*,*) 'balance call...'

c---- set pitch trim by adjusting CLh
      Wf = WTO - Wzero
      rfuel = Wf/parg(igWfuel)
      itrim = 1
      call balance(pari,parg,para(1,ip), rfuel,rpay,xipay, 
     &             itrim)
      CLh2 = para(iaCLh,ip)
      xCG2 = para(iaxCG,ip)
      xCP2 = para(iaxCP,ip)
      xNP2 = para(iaxNP,ip)
      do ip = 1, iprotate
        para(iaxCG,ip) = xCG2
        para(iaxCP,ip) = xCP2
        para(iaxNP,ip) = xNP2
        para(iaCLh,ip) = 0.
      enddo
c
c---- initial guesses for climb points
      ip = ipcutback
      pare(ieu0    ,ip) = V2
      pare(ieM0    ,ip) = M2
      para(iaMach  ,ip) = M2
      para(iaReunit,ip) = V2*pare(ierho0,ip) / pare(iemu0 ,ip)
      para(iaCL,ip) = CL2
      para(iaxCG,ip) = xCG2
      para(iaxCP,ip) = xCP2
      para(iaxNP,ip) = xNP2
      para(iaCLh,ip) = CLh2
c
      ip = ipclimb1
      pare(ieu0    ,ip) = V2
      pare(ieM0    ,ip) = M2
      para(iaMach  ,ip) = M2
      para(iaReunit,ip) = V2*pare(ierho0,ip) / pare(iemu0 ,ip)
      para(iaCL,ip) = CL2
      para(iaxCG,ip) = xCG2
      para(iaxCP,ip) = xCP2
      para(iaxNP,ip) = xNP2
      para(iaCLh,ip) = CLh2
c
c---- also use V2 speed for end-of-descent condition, with weight correction
      ip = ipdescentn
      Vrat = sqrt(para(iafracW,ip)/para(iafracW,ipclimb1))
      pare(ieu0    ,ip) = V2*Vrat
      pare(ieM0    ,ip) = M2*Vrat
      para(iaMach  ,ip) = M2*Vrat
      para(iaReunit,ip) = V2*Vrat*pare(ierho0,ip) / pare(iemu0 ,ip)
      para(iaCL,ip) = CL2
c
c=============================================================================
c---- set up climb points at equal altitude intervals, from altb to altc
c-    (takeoff ground temperature is neglected here -- std atmosphere is used)
      altb = para(iaalt,iptakeoff)
      altc = para(iaalt,ipcruise1)
      do ip = ipclimb1+1, ipclimbn
        frac = float(ip-ipclimb1) / float(ipclimbn-ipclimb1)
        para(iaalt,ip) = altb*(1.0-frac) + altc*frac
        altkm = para(iaalt,ip)/1000.0
        call atmos(altkm, T0,p0,rho0,a0,mu0)
        pare(iep0  ,ip) = p0
        pare(ieT0  ,ip) = T0
        pare(iea0  ,ip) = a0
        pare(ierho0,ip) = rho0
        pare(iemu0 ,ip) = mu0

        rhocab = max( parg(igpcabin) , p0 ) / (RSL*TSL)
        para(iaWbuoy,ip) = (rhocab-rho0)*gee*parg(igcabVol)
      enddo

c---- set climb Tt4's from fractions
      fT1 = parg(igfTt4CL1)
      fTn = parg(igfTt4CLn)
      Tt4TO = pare(ieTt4,iptakeoff)
      Tt4CR = pare(ieTt4,ipcruise1)
      do ip = ipclimb1, ipclimbn
        frac = float(ip      -ipclimb1)
     &       / float(ipclimbn-ipclimb1)
        Tfrac = fT1*(1.0-frac) + fTn*frac
        pare(ieTt4,ip) = Tt4TO*(1.0-Tfrac) + Tt4CR*Tfrac
      enddo

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- initial values for range, time, weight fraction
      ip = iprotate
      para(iaRange,ip) = 0.0
      para(iatime ,ip) = 0.0
      para(iafracW,ip) = WTO/WMTO
      para(iaWbuoy,ip) = 0.

      ip = iptakeoff
      para(iaRange,ip) = 0.0
      para(iatime ,ip) = 0.0
      para(iafracW,ip) = WTO/WMTO
      para(iaWbuoy,ip) = 0.

      ip = ipcutback
      para(iaRange,ip) = 0.0
      para(iatime ,ip) = 0.0
      para(iafracW,ip) = WTO/WMTO
      para(iaWbuoy,ip) = 0.

      ip = ipclimb1
      para(iaRange,ip) = 0.0
      para(iatime ,ip) = 0.0
      para(iafracW,ip) = WTO/WMTO
      para(iaWbuoy,ip) = 0.

      if(Ldebug) write(*,*) 'integrating climb...'

c---- integrate trajectory over climb
      do 20 ip = ipclimb1, ipclimbn
c------ velocity calculation from CL, Weight, altitude
        W  = para(iafracW,ip)*WMTO
        CL = para(iaCL,ip)
        rho = pare(ierho0,ip)
        mu = pare(iemu0,ip)
        Vsound = pare(iea0,ip)
        cosg = cos(para(iagamV,ip))
        BW = W + para(iaWbuoy,ip)

c------ iterate to converge climb angle gamV
        do 10 iterg = 1, itergmax
          V = sqrt(2.0*BW*cosg/(rho*S*CL))
          Mach = V/Vsound

          para(iaMach,ip) = Mach
          para(iaReunit,ip) = V*rho/mu

          pare(ieu0,ip) = V
          pare(ieM0,ip) = Mach

c-------- set pitch trim by adjusting CLh
          Wf = W - Wzero
          rfuel = Wf/parg(igWfuel)
          itrim = 1
          call balance(pari,parg,para(1,ip),rfuel,rpay,xipay, 
     &                 itrim)

          if(ip.eq.ipclimb1) then
           icdfun = 0  ! use explicitly specified wing cdf,cdp
          else
           icdfun = 1  ! use airfoil database # iairf for wing cdf,cdp
          endif
          call cdsum(pari,parg,para(1,ip),pare(1,ip), icdfun,iairf)

          icall = 1  ! use fixed engine geometry, specified Tt4
          icool = 1  ! use previously-set turbine cooling mass flow
          call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &                icall,icool,initeng,
     &                ichoke5(ip),ichoke7(ip)) 

          F    = pare(ieFe,ip)*parg(igneng)
          TSFC = pare(ieTSFC,ip)
          DoL = para(iaCD,ip)/para(iaCL,ip)

c-------- calculate improved flight angle gamV
          phi = F/BW
          sing = (phi - DoL*sqrt(1.0-phi**2 + DoL**2)) / (1.0+DoL**2)
          cosg = sqrt(1.0 - sing**2)
          gamV = atan2(sing,cosg)

          dgamV = gamV - para(iagamV,ip)

          para(iagamV,ip) = gamV

          if(abs(dgamV) .lt. gamVtol) go to 11
 10     continue
        write(*,*) 'MISSION: gamV not converged. dgamV =', dgamV
 11     continue

        if(Lengwrt) call engwrt(6,cplab(ip),pare(1,ip))

c------ store integrands for Range and Weight integration
        FoW(ip) = F/(BW*cosg) - DoL
        FFC(ip) = F/(W*V*cosg) * TSFC
        Vgi(ip) = 1.0/(V*cosg)

        Mach = para(iaMach,ip)
        CL = para(iaCL,ip)
        CD = para(iaCD,ip)
        gamV = para(iagamV,ip)

        if(Ldebug) then
         write(*,'( a,f9.5, f10.0, 2f10.5, 2f10.5,f10.5,f10.5,3g13.5)')
     &   ' climb :  fW W F/W D/L M CL CD gam Tt4',
     &     para(iafracW,ip), W, F/W, DoL, Mach, CL, CD, gamV*180.0/pi,
     &     pare(ieTt4,ip)
!     &     FoW(ip), FFC(ip), VGi(ip)

c      fBLIf = parg(igfBLIf)
c      dCDBLIf = -fBLIf*para(iaDAfwake,ip)/parg(igS)
c      write(*,'(1x,20f10.6)')
c     &  para(iaCL,ip),
c     &  para(iaCDi,ip),
c     &  para(iaCDfuse,ip),
c     &  para(iaCDwing,ip),
c     &  para(iaCDhtail,ip),
c     &  para(iaCDvtail,ip),
c     &  para(iaCDnace,ip),
c     &  para(iaCDover,ip),
c     &  dCDBLIf

        endif

        if(ip .gt. ipclimb1) then
c------- corrector integration step, approximate trapezoidal
         dh   = para(iaalt,ip) - para(iaalt,ip-1)
         dVsq = pare(ieu0,ip)**2 - pare(ieu0,ip-1)**2 

         FoWavg = 0.5*(FoW(ip) + FoW(ip-1))
         FFCavg = 0.5*(FFC(ip) + FFC(ip-1))
         Vgiavg = 0.5*(Vgi(ip) + Vgi(ip-1))
         dR = (dh + 0.5*dVsq/gee) / FoWavg
         dt = dR*Vgiavg
         rW = exp(-dR*FFCavg)

         para(iaRange,ip) = para(iaRange,ip-1) + dR
         para(iatime ,ip) = para(iatime ,ip-1) + dt
         para(iafracW,ip) = para(iafracW,ip-1)*rW
        endif

        if(ip .lt. ipclimbn) then
c------- predictor integration step, forward Euler
         if(para(iagamV,ip+1) .le. 0.0) then
c-------- if gamV guess is not passed in, use previous point as the guess
          para(iagamV,ip+1) = para(iagamV,ip)
         endif

         W  = para(iafracW,ip+1)*WMTO
         CL = para(iaCL,ip+1)
         rho = pare(ierho0,ip+1)
         cosg = cos(para(iagamV,ip+1))

         BW = W + para(iaWbuoy,ip+1)

         V = sqrt(2*BW*cosg/(rho*S*CL))
         pare(ieu0,ip+1) = V
         dh   = para(iaalt,ip+1) - para(iaalt,ip)
         dVsq = pare(ieu0,ip+1)**2 - pare(ieu0,ip)**2 

         dR = (dh + 0.5*dVsq/gee) / FoW(ip)
         dt = dR*Vgi(ip)
         rW = exp(-dR*FFC(ip))

         para(iaRange,ip+1) = para(iaRange,ip) + dR
         para(iatime ,ip+1) = para(iatime ,ip) + dt
         para(iafracW,ip+1) = para(iafracW,ip)*rW
        endif

c        qinf = 0.5*rho*V**2
c        write(*,'(a,5g13.5)')' Sb:', WMTO, para(iafracW,ip), qinf, S, CL

c        write(*,*)
c        write(*,*) 'dh km =', dh / 1000.0
c        write(*,*) 'dVsq  =', dVsq
c        write(*,*) 'dR km =', dR / 1000.0
c        write(*,*) 'dt min=', dt / 60.0
c        write(*,*) 'rWbc  =', rW 

 20   continue


      if(Ldebug) write(*,*) '...done integrating climb'

c-------------------------------------------------------------------
c---- first point in cruise is last point in climb
      para(iaRange,ipcruise1) = para(iaRange,ipclimbn)
      para(iatime ,ipcruise1) = para(iatime ,ipclimbn)
      para(iafracW,ipcruise1) = para(iafracW,ipclimbn)
      para(iaWbuoy,ipcruise1) = para(iaWbuoy,ipclimbn)

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- first cruise point
      ip = ipcruise1
c
c---- set pitch trim by adjusting CLh
      Wf = para(iafracW,ip)*WMTO - Wzero
      rfuel = Wf/parg(igWfuel)
      itrim = 1
c      if(ip.eq.10) write(*,*) 'MISSION 2:', rfuel,rpay
      call balance(pari,parg,para(1,ip),rfuel,rpay,xipay, 
     &             itrim)

      if(ipc1 .eq. 0) then
c----- calculate cruise1 point only if requested
       icdfun = 1  ! use airfoil database for wing airfoil cdf,cdp
       call cdsum(pari,parg,para(1,ip),pare(1,ip), icdfun,iairf)
       DoL = para(iaCD,ip)/para(iaCL,ip)
       W = para(iafracW,ip)*WMTO
       BW = W + para(iaWbuoy,ip)
       F = BW * (DoL + para(iagamV,ip))
c       pare(ieFe,ip) = F/parg(igneng)

       icall = 2  ! use fixed engine geometry, specified Fe
       icool = 1  ! use previously-set turbine cooling mass flow
       call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &             icall,icool,initeng,
     &             ichoke5(ip),ichoke7(ip)) 
      endif

c---- set cruise-climb climb angle, from fuel burn rate and atmospheric dp/dz
      TSFC = pare(ieTSFC,ip)
      V    = pare(ieu0  ,ip)
      p0   = pare(iep0  ,ip)
      rho0 = pare(ierho0,ip)
      DoL  = para(iaCD,ip)/para(iaCL,ip)
      gamVcr1 = DoL*p0*TSFC/(rho0*gee*V - p0*TSFC)

      F    = pare(ieFe,ip)*parg(igneng)
      W    = para(iafracW,ip)*WMTO
      cosg = cos(gamVcr1)

      BW = W + para(iaWbuoy,ip)

      FoW(ip) = F/(BW*cosg) - DoL
      FFC(ip) = F/(W*V*cosg) * TSFC
      Vgi(ip) = 1.0/(V*cosg)

      if(Ldebug) then
       write(*,'(a,f9.5, f10.0, 2f11.6, 2f11.6, f11.7, f10.6, 5g13.5)')
     &   ' beg CR:  fW W F/W D/L M CL CD gam FoW FFC VGi F gam',
     &     para(iafracW,ip), W, F/W, DoL, Mach,
     &     para(iaCL,ip), para(iaCD,ip), para(iagamv,ip)*180.0/pi,
     &     FoW(ip), FFC(ip), Vgi(ip), F, gamVcr1*180.0/pi
      endif

      para(iagamV,ip) = gamVcr1

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- set end-of-cruise point "d" using cruise and descent angles, 
c-     and remaining range legs
      gamVde1 = parm(imgamVDE1)
      gamVden = parm(imgamVDEn)
      gamVdeb = 0.5*(gamVde1+gamVden)
      alte = para(iaalt,ipdescentn)
      dRclimb  = para(iaRange,ipclimbn)
     &         - para(iaRange,ipclimb1)
      dRcruise = (alte - altc - gamVdeb*(Rangetot-dRclimb))
     &         / (gamVcr1 - gamVdeb)

      altd = altc + gamVcr1*dRcruise

c      write(*,'(1x, 20f12.7)')
c     &   dRcruise/4.69e6,
c     &   altc/1e3,
c     &   altd/1e3,
c     &   alte/1e3,
c     &   gamVde1*1000,
c     &   gamVcr1*1000,
c     &   Rangetot/5e6,
c     &   dRclimb/5e4

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- last cruise point
      ip = ipcruisen
      Mach = para(iaMach,ip)
      altkm = altd/1000.0
      call atmos(altkm, T0,p0,rho0,a0,mu0)
      pare(iep0  ,ip) = p0
      pare(ieT0  ,ip) = T0
      pare(iea0  ,ip) = a0
      pare(ierho0,ip) = rho0
      pare(iemu0 ,ip) = mu0
      pare(ieM0  ,ip) = Mach
      pare(ieu0  ,ip) = Mach*a0
      para(iaReunit,ip) = Mach*a0 * rho0/mu0
      para(iaalt ,ip) = altd

      rhocab = max( parg(igpcabin) , p0 ) / (RSL*TSL)
      para(iaWbuoy,ip) = (rhocab-rho0)*gee*parg(igcabVol)

c---- set pitch trim by adjusting CLh
      Wf = para(iafracW,ip)*WMTO - Wzero
      rfuel = Wf/parg(igWfuel)
      itrim = 1
      call balance(pari,parg,para(1,ip),rfuel,rpay,xipay, 
     &             itrim)

      icdfun = 1  ! use airfoil database for wing airfoil cdf,cdp
      call cdsum(pari,parg,para(1,ip),pare(1,ip), icdfun,iairf)
      DoL = para(iaCD,ip)/para(iaCL,ip)
      W = para(iafracW,ip)*WMTO
      BW = W + para(iaWbuoy,ip)
      F = BW * (DoL + para(iagamV,ip))
      pare(ieFe,ip) = F/parg(igneng)

      icall = 2  ! use fixed engine geometry, specified Fe
      icool = 1  ! use previously-set turbine cooling mass flow
      call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &            icall,icool,initeng,
     &            ichoke5(ip),ichoke7(ip)) 

      TSFC = pare(ieTSFC,ip)
      V    = pare(ieu0  ,ip)
      p0   = pare(iep0  ,ip)
      rho0 = pare(ierho0,ip)
      DoL  = para(iaCD,ip)/para(iaCL,ip)
      gamVcr2 = DoL*p0*TSFC/(rho0*gee*V - p0*TSFC)

c---- note: F/W is assumed unchanged from first cruise point, 
c-    since W is not yet known (could be iterated)
      cosg = cos(gamVcr2)

      FoW(ip) = F/(BW*cosg) - DoL
      FFC(ip) = F/(W*V*cosg) * TSFC
      Vgi(ip) = 1.0/(V*cosg)

      if(Ldebug) then
       write(*,'(a,f9.5, f10.0, 2f11.6, 2f11.6, f11.7, f10.6, 5g13.5)')
     &   ' end CR:  fW W F/W D/L M CL CD gam FoW FFC VGi F gam',
     &     para(iafracW,ip), W, F/W, DoL, Mach,
     &     para(iaCL,ip), para(iaCD,ip), para(iagamv,ip)*180.0/pi,
     &     FoW(ip), FFC(ip), Vgi(ip), F, gamVcr2*180.0/pi
      endif

      para(iagamV,ip) = gamVcr2

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- set changes over cruise with single integration interval,
c-    using the same assumption as Breguet (constant integrand)
      ip1 = ipcruise1
      ipn = ipcruisen

      FoWavg = 0.5*(FoW(ipn) + FoW(ip1))
      FFCavg = 0.5*(FFC(ipn) + FFC(ip1))
      Vgiavg = 0.5*(Vgi(ipn) + Vgi(ip1))

cc    dVsq = pare(ieu0,ip)**2 - pare(ieu0,ip-1)**2 
cc    dRcruise = (dh + 0.5*dVsq/gee) / FoWavg
      dtcruise = dRcruise*Vgiavg
      rWcruise = exp(-dRcruise*FFCavg)

      para(iaRange,ipn) = para(iaRange,ip1) + dRcruise
      para(iatime ,ipn) = para(iatime ,ip1) + dtcruise
      para(iafracW,ipn) = para(iafracW,ip1)*rWcruise

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- set intermediate points over cruise, if any, just by interpolating
      do ip = ipcruise1+1, ipcruisen-1
        frac = float(ip       -ipcruise1)
     &       / float(ipcruisen-ipcruise1)
        Mach = para(iaMach,ip)
        para(iaalt,ip) = altc*(1.0-frac) + altd*frac
        altkm = para(iaalt,ip)/1000.0
        call atmos(altkm, T0,p0,rho0,a0,mu0)
        pare(iep0  ,ip) = p0
        pare(ieT0  ,ip) = T0
        pare(iea0  ,ip) = a0
        pare(ierho0,ip) = rho0
        pare(iemu0 ,ip) = mu0
        pare(ieM0  ,ip) = Mach
        pare(ieu0  ,ip) = Mach*a0
        para(iaReunit,ip) = Mach*a0 * rho0/mu0

        rhocab = max( parg(igpcabin) , p0 ) / (RSL*TSL)
        para(iaWbuoy,ip) = (rhocab-rho0)*gee*parg(igcabVol)

        para(iaRange,ip) = para(iaRange,ipcruise1) + dRcruise * frac
        para(iatime ,ip) = para(iatime ,ipcruise1) + dtcruise * frac
        para(iafracW,ip) = para(iafracW,ipcruise1) * rWcruise ** frac
      enddo

c=============================================================================
c---- first descent point is same as last cruise point computed above
      ip = ipdescent1
      pare(iep0  ,ip)  = pare(iep0  ,ipcruisen)  
      pare(ieT0  ,ip)  = pare(ieT0  ,ipcruisen)  
      pare(iea0  ,ip)  = pare(iea0  ,ipcruisen)  
      pare(ierho0,ip)  = pare(ierho0,ipcruisen)  
      pare(iemu0 ,ip)  = pare(iemu0 ,ipcruisen)  
      pare(ieM0  ,ip)  = pare(ieM0  ,ipcruisen)  
      pare(ieu0  ,ip)  = pare(ieu0  ,ipcruisen)  

      para(iaMach  ,ip) = para(iaMach  ,ipcruisen)
      para(iaReunit,ip) = para(iaReunit,ipcruisen)
      para(iaalt   ,ip) = para(iaalt   ,ipcruisen)

      para(iaRange,ip) = para(iaRange,ipcruisen)
      para(iatime ,ip) = para(iatime ,ipcruisen)
      para(iafracW,ip) = para(iafracW,ipcruisen)
      para(iaWbuoy,ip) = para(iaWbuoy,ipcruisen)

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- set up remaining descent points at equal distance intervals,
c-    assuming a linearly-varying descent slope
      Rd = para(iaRange,ipdescent1)
      Re = para(iaRange,ipdescentn)
      altd = para(iaalt,ipdescent1)
      alte = para(iaalt,ipdescentn)
      para(iagamV,ipdescent1) = gamVde1
      para(iagamV,ipdescentn) = gamVden
      do ip = ipdescent1+1, ipdescentn-1
        frac = float(ip        -ipdescent1)
     &       / float(ipdescentn-ipdescent1)
        R   = Rd  *(1.0-frac) + Re  *frac
ccc     alt = altd*(1.0-frac) + alte*frac
        alt = altd + (Re-Rd)*( gamVde1*(frac-0.5*frac**2)
     &                       + gamVden*      0.5*frac**2  )
        gamVde = gamVde1*(1.0-frac)
     &         + gamVden*     frac
        para(iagamV,ip) = gamVde

        altkm = alt/1000.0
        call atmos(altkm, T0,p0,rho0,a0,mu0)
        pare(iep0  ,ip) = p0
        pare(ieT0  ,ip) = T0
        pare(iea0  ,ip) = a0
        pare(ierho0,ip) = rho0
        pare(iemu0 ,ip) = mu0

        para(iaRange,ip) = R
        para(iaalt  ,ip) = alt

        rhocab = max( parg(igpcabin) , p0 ) / (RSL*TSL)
        para(iaWbuoy,ip) = (rhocab-rho0)*gee*parg(igcabVol)
      enddo
      para(iaWbuoy,ipdescentn) = 0.

c---- integrate time and weight over descent
      do 80 ip = ipdescent1, ipdescentn

c------ velocity calculation from CL, Weight, altitude
        gamVde = para(iagamV,ip)
        cosg = cos(gamVde)
        W  = para(iafracW,ip)*WMTO
        BW = W + para(iaWbuoy,ip)
        CL = para(iaCL,ip)
        rho = pare(ierho0,ip)
        V = sqrt(2.0*BW*cosg/(rho*S*CL))
        Mach = V/pare(iea0,ip)

        para(iaMach,ip) = Mach
        para(iaReunit,ip) = V*rho/pare(iemu0,ip)

        pare(ieu0,ip) = V
        pare(ieM0,ip) = Mach

c------ set pitch trim by adjusting CLh
        Wf = W - Wzero
        rfuel = Wf/parg(igWfuel)
        itrim = 1
        call balance(pari,parg,para(1,ip), rfuel,rpay,xipay, 
     &               itrim)
        
        if(ip.eq.ipdescentn) then
         icdfun = 0  ! use explicitly specified wing cdf,cdp
        else
         icdfun = 1  ! use airfoil database for wing cdf,cdp
        endif
        call cdsum(pari,parg,para(1,ip),pare(1,ip), icdfun,iairf)

c------ set up for engine calculation
        sing = sin(gamVde)
        cosg = cos(gamVde)
        DoL = para(iaCD,ip)/para(iaCL,ip)
        Fspec = BW*(sing + cosg*DoL)
        pare(ieFe,ip) = Fspec / parg(igneng)

        if(initeng.eq.0) then
c------- engine states are assumed to be NOT initialized

c------- set to zero to force initialization
c        pare(iembf ,ip) = 0. ! pare(iembfD ,ip)
c        pare(iemblc,ip) = 0. ! pare(iemblcD,ip)
c        pare(iembhc,ip) = 0. ! pare(iembhcD,ip)
c        pare(iepif ,ip) = 0. ! pare(iepifD ,ip)
c        pare(iepilc,ip) = 0. ! pare(iepilcD,ip)
c        pare(iepihc,ip) = 0. ! pare(iepihcD,ip)
c        inite = 0

c------- use previous point to initialize engine state for prescribed thrust
         pare(iembf ,ip) = pare(iembf ,ip-1)
         pare(iemblc,ip) = pare(iemblc,ip-1)
         pare(iembhc,ip) = pare(iembhc,ip-1)
         pare(iepif ,ip) = pare(iepif ,ip-1)
         pare(iepilc,ip) = pare(iepilc,ip-1)
         pare(iepihc,ip) = pare(iepihc,ip-1)
         inite = 1

c         if(ip.eq.ipdescent1) then
c          Frat = pare(ieFe,ip)/pare(ieFe,ip-1)
c          pare(iepif,ip) = (pare(iepif,ip-1)-1.0)*Frat + 1.0
c          write(*,*) Frat, pare(iepif,ip-1), pare(iepif,ip)
c         endif

c------- make better estimate for new Tt4, adjusted for new ambient T0
         dTburn = pare(ieTt4,ip-1) - pare(ieTt3,ip-1)
         OTR = pare(ieTt3,ip-1)/pare(ieTt2,ip-1)
         Tt3 = pare(ieT0,ip) * OTR
         pare(ieTt4,ip) = Tt3 + dTburn + 50.0

c------- make better estimate for new pt5, adjusted for new ambient p0
         pare(iept5,ip) = pare(iept5,ip-1)
     &                  * pare(iep0,ip)
     &                  / pare(iep0,ip-1)
        else
         inite = initeng

        endif

        icall = 2  ! use fixed engine geometry, specified Fe
        icool = 1  ! use previously-set turbine cooling mass flow
        call tfcalc(pari,parg,para(1,ip),pare(1,ip), ip,
     &              icall,icool,inite,
     &              ichoke5(ip),ichoke7(ip)) 
        if(Lengwrt) call engwrt(6,cplab(ip),pare(1,ip))

c------ store effective thrust, effective TSFC
        F    = pare(ieFe,ip)*parg(igneng)
        TSFC = pare(ieTSFC,ip)

c------ store integrands for Range and Weight integration
        FoW(ip) = F/(BW*cosg) - DoL
        FFC(ip) = F/(W*V*cosg) * TSFC
        Vgi(ip) = 1.0/(V*cosg)

c------ if F < 0, then TSFC is not valid, so calculate mdot_fuel directly
        mfuel = pare(ieff,ip)*pare(iemcore,ip) * parg(igneng)
        FFC(ip) = gee*mfuel / (W*cosg*V)

      if(Ldebug) then
       write(*,'(a,f9.5, f10.0, 2f11.6, 2f10.5, f10.5, f10.5, 5g13.5)')
     &   ' desc  :  fW W F/W D/L M CL CD gam FoW FFC VGi F DoL',
     &     para(iafracW,ip), W, F/W, DoL, Mach,
     &     para(iaCL,ip), para(iaCD,ip), gamVcr1*180.0/pi,
     &     FoW(ip), FFC(ip), Vgi(ip), F, DoL
      endif

        if(ip .gt. ipdescent1) then
c------- corrector integration step, approximate trapezoidal
         dh   = para(iaalt,ip) - para(iaalt,ip-1)
         dVsq = pare(ieu0,ip)**2 - pare(ieu0,ip-1)**2 

         FoWavg = 0.5*(FoW(ip) + FoW(ip-1))
         FFCavg = 0.5*(FFC(ip) + FFC(ip-1))
         Vgiavg = 0.5*(Vgi(ip) + Vgi(ip-1))

ccc      dR = (dh + 0.5*dVsq/gee) / FoWavg
         dR = para(iaRange,ip) - para(iaRange,ip-1)
         dt = dR*Vgiavg
         rW = exp(-dR*FFCavg)

ccc      para(iaRange,ip) = para(iaRange,ip-1) + dR
         para(iatime ,ip) = para(iatime ,ip-1) + dt
         para(iafracW,ip) = para(iafracW,ip-1)*rW
        endif

        if(ip .lt. ipdescentn) then
c------- predictor integration step, forward Euler
         gamVde = para(iagamV,ip+1)
         cosg = cos(gamVde)
         W  = para(iafracW,ip+1)*WMTO

         BW = W + para(iaWbuoy,ip+1)

         CL = para(iaCL,ip+1)
         rho = pare(ierho0,ip+1)
         V = sqrt(2*BW*cosg/(rho*S*CL))
         pare(ieu0,ip+1) = V

         dh   = para(iaalt,ip+1) - para(iaalt,ip)
         dVsq = pare(ieu0,ip+1)**2 - pare(ieu0,ip)**2 

ccc      dR = (dh + 0.5*dVsq/gee) / FoW(ip)
         dR = para(iaRange,ip) - para(iaRange,ip-1)
         dt = dR*Vgi(ip)
         rW = exp(-dR*FFC(ip))

ccc      para(iaRange,ip+1) = para(iaRange,ip) + dR
         para(iatime ,ip+1) = para(iatime ,ip) + dt
         para(iafracW,ip+1) = para(iafracW,ip)*rW
        endif
 80   continue

c=====================================================================
c---- mission fuel fractions and weights
      fracWa = para(iafracW,ipclimb1)
      fracWe = para(iafracW,ipdescentn)
      freserve = parg(igfreserve)
      fburn = fracWa - fracWe
      ffuel = fburn*(1.0+freserve)
      Wfuel  = WMTO*ffuel
      WTO = Wzero + Wfuel

      parm(imWTO  ) = WTO
      parm(imWfuel) = Wfuel

c---- dump the complete output state
      open(87,file='mission_out.txt',status='unknown')
      do k = 1, iatotal
        write(87,'(e26.18)') (para(k,ipx), ipx=1,iptotal)
      enddo
      do k = 1, ietotal
        write(87,'(e26.18)') (pare(k,ipx), ipx=1,iptotal)
      enddo
      write(87,'(e26.18)') WTO, Wfuel, fburn
      close(87)
c---- mission PFEI
      Wburn = WMTO*fburn
      parm(imPFEI) = Wburn*pare(iehfuel,ipcruise1)
     &           / (parm(imWpay)*parm(imRange))

c      write(*,*) 'exiting MISSION...'

      return
      end ! mission
