
      subroutine aswout(lu,pari,parg,para,configname)
      implicit real (a-h,m,o-z)

      include 'index.inc'
      integer lu
      integer pari(iitotal)
      real parg(igtotal),
     &     para(iatotal)
      character*(*) configname

      include 'fbl.inc'
      include 'constants.inc'

C------------------------------------------------------------------------------
C---- ASWING dimensions and arrays
      PARAMETER (
     & IBX = 100,   ! number of specified-quantity points along beam
     & NBX = 35,    ! number of beams  (beam = surface or fuselage)     
     & NJX = 30,    ! number of beam joints      
     & NGX = 8,     ! number of beam grounds (kinematic fixing locations)
     & NPX = 50,    ! number of pylons (point masses+struts+engines+sensors)
     & NNX = 200,   ! number of circulation harmonics for all surfaces 
     & NAJX = 40,   ! number of points defining  angle(Moment) joint curve
     & NSENX = 12 ) ! number of sensors                                   
      PARAMETER (
     &  NFLPX = 20,   ! number of flap variables    ( see note above! )
     &  NENGX = 12 )  ! number of engine variables  ( see note above! )
C
C- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C  NOTE:  JBX must be at least as big as JBTOT in INDEXB.INC
      PARAMETER ( 
     & KPX = 21,   ! number of quantities defining pylon
     & JBX = 102)  ! number of beam-defining variables  (see note above!)
C

      LOGICAL LQBDEF(JBX,NBX)
      LOGICAL LBSYMM(JBX,NBX)
C
      INTEGER NB(0:JBX,NBX), KBNUM(NBX), IBEAM(NBX)
      REAL QB(IBX,0:JBX,NBX)
      REAL TB(IBX,0:JBX,NBX)
C
      REAL TJOIN(2,NJX)
      INTEGER KBJOIN(2,NJX), KJTYPE(NJX)
C
      REAL TGROU(NGX)
      INTEGER KBGROU(NGX), KGTYPE(NGX)
C
      REAL QPYLO(0:KPX,NPX)
      INTEGER KBPYLO(NPX), KPTYPE(NPX)
      INTEGER IENGTYP(NENGX)
      REAL XYZREF(3,3)
C
      INTEGER NANGJ(NJX)
      REAL MOMJ(NAJX,NJX), ANGJ(NAJX,NJX), HJAX(3,NJX)
C
      CHARACTER*32 UNCHL, UNCHM, UNCHT, UNCHF
      CHARACTER*64 BNAME(NBX)


      real Nsho, Nshs, Nsht, Nshh, Nshv, Nsh
      logical Lttail, Lptail


      INCLUDE 'INDEXB.INC'

      

c---- INSTRUMENTATION: dump the complete input state on entry, so the port
c---- can be handed exactly what this call received. Not part of TASOPT --
c---- see fortran_ref/aswout_instrumented.f.
      open(87,file='aswout_in.txt',status='unknown')
      write(87,'(i6)') (kdum, kdum=1,1)
      write(87,'(i6)') (pari(kdum), kdum=1,iitotal)
      write(87,'(e26.18)') (parg(kdum), kdum=1,igtotal)
      write(87,'(e26.18)') (para(kdum), kdum=1,iatotal)
      write(87,'(i6)') nbl, iblte
      write(87,'(e26.18)') (xbl(kdum), kdum=1,nbl)
      write(87,'(e26.18)') (uebl(kdum), kdum=1,nbl)
      write(87,'(e26.18)') (thbl(kdum), kdum=1,nbl)
      write(87,'(e26.18)') (tsbl(kdum), kdum=1,nbl)
      write(87,'(e26.18)') (cdbl(kdum), kdum=1,nbl)
      write(87,'(a)') configname
      close(87)

      Lttail = .false.
      Lptail = parg(ignvtail) .gt. 1.0001

      xnose = parg(igxnose)
      xend  = parg(igxend )
      xwbox = parg(igxwbox)

      xhbox = parg(igxhbox)
      xvbox = parg(igxvbox)

      xblend1 = parg(igxblend1)
      xblend2 = parg(igxblend2)

      xshell1 = parg(igxshell1)
      xshell2 = parg(igxshell2)
      xconend = parg(igxconend)

      ifwcen = pari(iifwcen)
      iwplan = pari(iiwplan)

      iengloc = pari(iiengloc)

      CLMf0 = parg(igCLMf0)

      if(CLMf0 .lt. 0.0) then
c----- positive nose-up moment... assume upturned nose
       xupsweep = xnose + 0.75*(xwbox-xnose)
       dupsweep = 0.7*(parg(igRfuse) + 0.5*parg(igdRfuse))
      else
       xupsweep = xnose + 0.75*(xwbox-xnose)
       dupsweep = -0.2*parg(igRfuse)
      endif

      if(Lptail) then
       yov = parg(igwfb) + parg(igRfuse)
       ytv = 0.5*parg(igboh)
       zov = 0.5*parg(igbov)
       ztv =     parg(igbv)
       t0v = 0.
       ttv = yov + ztv
      else
       yov = 0.
       ytv = 0.
       zov = 0.5*parg(igbov)
       ztv =     parg(igbv)
       t0v = 1.0
       ttv = ztv
      endif

      EAfac = 0.001
      EAfac = 1.0

c================================================================

      do IS = 1, NBX
        do J = 1, JBX
          LQBDEF(J,IS) = .false.
          LBSYMM(J,IS) = .false.
        enddo
      enddo

      do KP = 0, KPX
        do NP = 1, NPX
          QPYLO(KP,NP) = 0.
        enddo
      enddo


      UNCHL = 'm'
      UNCHM = 'kg'
      UNCHT = 's'
      UNCHF = 'N'
      UNITL = 1.0
      UNITM = 1.0
      UNITT = 1.0
      UNITF = 1.0

      SREF = parg(igS)
      BREF = parg(igb)
      CREF = parg(igcma)

      DO L = 1, 3
        XYZREF(1,L) = xwbox
        XYZREF(2,L) = 0.
        XYZREF(3,L) = 0.
      ENDDO

      NBEAM = 0
      NJOIN = 0
      NGROU = 0
      NPYLO = 0
      NANGJ = 0

c================================================================
c---- ground
      NGROU = NGROU + 1
      TGROU(NGROU) = xwbox
      KBGROU(NGROU) = 1
      KGTYPE(NGROU) = 0

c      NGROU = NGROU + 1
c      TGROU(NGROU) = 0.
c      KBGROU(NGROU) = 2
c      KGTYPE(NGROU) = 0

      NGROU = NGROU + 1
      TGROU(NGROU) = 0.5*parg(igbo)
      KBGROU(NGROU) = 2
      KGTYPE(NGROU) = 0

      NGROU = NGROU + 1
      TGROU(NGROU) = -0.5*parg(igbo)
      KBGROU(NGROU) = 2
      KGTYPE(NGROU) = 0

c================================================================
c---- joints

cc---- fuselage to wing
c      NJOIN = NJOIN + 1
c      KBJOIN(1,NJOIN) = 1
c      KBJOIN(2,NJOIN) = 2
c      TJOIN(1,NJOIN) = xwbox
c      TJOIN(2,NJOIN) = 0.5*parg(igbo)
c      KJTYPE(NJOIN) = 0

cc---- fuselage to wing
c      NJOIN = NJOIN + 1
c      KBJOIN(1,NJOIN) = 1
c      KBJOIN(2,NJOIN) = 2
c      TJOIN(1,NJOIN) = xwbox
c      TJOIN(2,NJOIN) = -0.5*parg(igbo)
c      KJTYPE(NJOIN) = 0

      if(Lttail) then
c----- T-tail

c----- join fuselage to bottom of VT
       NJOIN = NJOIN + 1
       KBJOIN(1,NJOIN) = 1
       KBJOIN(2,NJOIN) = 4
       TJOIN(1,NJOIN) = xvbox
       TJOIN(2,NJOIN) = t0v
       KJTYPE(NJOIN) = 0

c----- join top of VT to center of HT
       NJOIN = NJOIN + 1
       KBJOIN(1,NJOIN) = 4
       KBJOIN(2,NJOIN) = 3
       TJOIN(1,NJOIN) = ttv
       TJOIN(2,NJOIN) = 0.
       KJTYPE(NJOIN) = 0

      elseif(Lptail) then
c----- Pi-tail

c----- join fuselage to bottom of VT
       NJOIN = NJOIN + 1
       KBJOIN(1,NJOIN) = 1
       KBJOIN(2,NJOIN) = 4
       TJOIN(1,NJOIN) = xvbox
       TJOIN(2,NJOIN) = t0v
       KJTYPE(NJOIN) = 0

c----- join top of right VT to right join point of HT
       NJOIN = NJOIN + 1
       KBJOIN(1,NJOIN) = 4
       KBJOIN(2,NJOIN) = 3
       TJOIN(1,NJOIN) = ttv
       TJOIN(2,NJOIN) = 0.5*parg(igboh)
       KJTYPE(NJOIN) = 0

c----- join top of left VT to left join point of HT
       NJOIN = NJOIN + 1
       KBJOIN(1,NJOIN) = 4
       KBJOIN(2,NJOIN) = 3
       TJOIN(1,NJOIN) = -ttv
       TJOIN(2,NJOIN) = -0.5*parg(igboh)
       KJTYPE(NJOIN) = 0

      else
c----- conventional tail

c----- join fuselage to bottom of VT
       NJOIN = NJOIN + 1
       KBJOIN(1,NJOIN) = 1
       KBJOIN(2,NJOIN) = 4
       TJOIN(1,NJOIN) = xvbox
       TJOIN(2,NJOIN) = t0v
       KJTYPE(NJOIN) = 0

c----- join fuselage to center of HT
       NJOIN = NJOIN + 1
       KBJOIN(1,NJOIN) = 1
       KBJOIN(2,NJOIN) = 3
       TJOIN(1,NJOIN) = xhbox
       TJOIN(2,NJOIN) = 0.
       KJTYPE(NJOIN) = 0

      endif

c================================================================
c---- weights

cc   &          KPX, NPX, NPYLO, QPYLO, KBPYLO, KPTYPE,
c    &   '      t    ',
c    &   '      Xp   ',
c    &   '      Yp   ',
c    &   '      Zp   ',
c    &   '      Mg   ',
c    &   '      CDA  ',
c    &   '      Vol  ',
c    &   '      Hxg  ',
c    &   '      Hyg  ',
c    &   '      Hzg  '


      NPYLO = NPYLO + 1
      KPTYPE(NPYLO) = 1
      KBPYLO(NPYLO) = 1
      QPYLO(0,NPYLO) = parg(igxfix)
      QPYLO(1,NPYLO) = parg(igxfix)
      QPYLO(2,NPYLO) = 0.
      QPYLO(3,NPYLO) = dupsweep
      QPYLO(4,NPYLO) = parg(igWfix)

      NPYLO = NPYLO + 1
      KPTYPE(NPYLO) = 1
      KBPYLO(NPYLO) = 1
      QPYLO(0,NPYLO) = parg(igxapu)
      QPYLO(1,NPYLO) = parg(igxapu)
      QPYLO(2,NPYLO) = 0.
      QPYLO(3,NPYLO) = 0.
      QPYLO(4,NPYLO) = parg(igWpay)*parg(igfapu)

      NPYLO = NPYLO + 1
      KPTYPE(NPYLO) = 1
      KBPYLO(NPYLO) = 1
      QPYLO(0,NPYLO) = parg(igxhpesys)
      QPYLO(1,NPYLO) = parg(igxhpesys)
      QPYLO(2,NPYLO) = 0.
      QPYLO(3,NPYLO) = -0.8*parg(igRfuse)
      QPYLO(4,NPYLO) = parg(igWMTO)*parg(igfhpesys)

      NPYLO = NPYLO + 1
      KPTYPE(NPYLO) = 1
      KBPYLO(NPYLO) = 1
      QPYLO(0,NPYLO) = parg(igxlgnose)
      QPYLO(1,NPYLO) = parg(igxlgnose)
      QPYLO(2,NPYLO) = 0.
      QPYLO(3,NPYLO) = -0.8*parg(igRfuse)
      QPYLO(4,NPYLO) = parg(igWMTO)*parg(igflgnose)


      xlgmain = parg(igxCGaft) + parg(igdxlgmain)
      ylgmain = 1.2 * (parg(igwfb) + parg(igRfuse))
      zlgmain = -0.8*(parg(igRfuse) + 0.5*parg(igdRfuse))

      NPYLO = NPYLO + 1
      KPTYPE(NPYLO) = 1
      KBPYLO(NPYLO) = 1
      QPYLO(0,NPYLO) = xwbox
      QPYLO(1,NPYLO) = xlgmain
      QPYLO(2,NPYLO) = ylgmain
      QPYLO(3,NPYLO) = zlgmain
      QPYLO(4,NPYLO) = 0.5 * parg(igWMTO)*parg(igflgmain)

      NPYLO = NPYLO + 1
      KPTYPE(NPYLO) = 1
      KBPYLO(NPYLO) = 1
      QPYLO(0,NPYLO) = xwbox
      QPYLO(1,NPYLO) = xlgmain
      QPYLO(2,NPYLO) = -ylgmain
      QPYLO(3,NPYLO) = zlgmain
      QPYLO(4,NPYLO) = 0.5 * parg(igWMTO)*parg(igflgmain)


      neng = int(parg(igneng)+0.001)



c================================================================
c---- engines

C------ write engine-data label line
c        WRITE(LU,1700) '#',
c     &   ' KPeng',
c     &   ' IEtyp',
c     &   ' Nbeam',
c     &   '       t      ',
c     &   '      Xp   ',
c     &   '      Yp   ',
c     &   '      Zp   ',
c     &   '      Tx   ',
c     &   '      Ty   ',
c     &   '      Tz   ',
c     &   '   dFdPeng ',
c     &   '   dMdPeng ',
c        ELSEIF( IPTYPE.EQ.3 .AND. KPTYPE(KP).GT.10      ) THEN
c          KE = KPTYPE(KP) - 10
c          IETYP = IENGTYP(KE)

      do ieng = 1, neng
        if(neng.eq.1) then
         frac = 0.0
        else
         frac = float(ieng-1)/float(neng-1)
        endif
        esgn = 1.0 - 2.0*frac
        xeng = parg(igxeng)
        yeng = parg(igyeng) * esgn

        NPYLO = NPYLO + 1
        KPTYPE(NPYLO) = 10 + ieng
        KE = ieng
        IENGTYP(KE) = 0

        QPYLO(1,NPYLO) = parg(igxeng)
        QPYLO(2,NPYLO) = yeng
        QPYLO(3,NPYLO) = 0.

        QPYLO(4,NPYLO) = -1.0
        QPYLO(5,NPYLO) = 0.
        QPYLO(6,NPYLO) = 0.

        QPYLO(7,NPYLO) = 1.0 / float(neng)
        QPYLO(8,NPYLO) = 0.

        if(iengloc .eq. 1) then
c------- wing engine
         KBPYLO(NPYLO) = 2
         QPYLO(0,NPYLO) = 0.5*parg(igbs) * esgn
         QPYLO(3,NPYLO) = parg(igzwing) - 0.3*parg(igco)*parg(iglambdas)
        else
c------- tail engine
         KBPYLO(NPYLO) = 1
         QPYLO(0,NPYLO) = parg(igxvbox)
         QPYLO(3,NPYLO) = 0.5*parg(igRfuse)
        endif
      enddo


c---- engine weights
      do ieng = 1, neng
        if(neng.eq.1) then
         frac = 0.0
        else
         frac = float(ieng-1)/float(neng-1)
        endif
        esgn = 1.0 - 2.0*frac
        xeng = parg(igxeng)
        yeng = parg(igyeng) * esgn

        NPYLO = NPYLO + 1
        KPTYPE(NPYLO) = 1

        QPYLO(1,NPYLO) = parg(igxeng)
        QPYLO(2,NPYLO) = yeng
        QPYLO(3,NPYLO) = 0.
        QPYLO(4,NPYLO) = parg(igWeng) / float(neng)

        if(iengloc .eq. 1) then
c------- wing engine
         KBPYLO(NPYLO) = 2
         QPYLO(0,NPYLO) = 0.5*parg(igbs) * esgn
         QPYLO(3,NPYLO) = parg(igzwing) - 0.3*parg(igco)*parg(iglambdas)
        else
c------- tail engine
         KBPYLO(NPYLO) = 1
         QPYLO(0,NPYLO) = parg(igxvbox)
         QPYLO(3,NPYLO) = 0.5*parg(igRfuse)
        endif

      enddo


c================================================================
c---- fuselage
      NBEAM = NBEAM + 1

      IS = NBEAM

      KBNUM(IS) = IS
      IBEAM(IS) = IS
      BNAME(IS) = 'Fuselage'


c---- fuselage shape parameters
      ifclose = pari(iifclose)
      wfb    = parg(igwfb)
      Rfuse  = parg(igRfuse)
      dRfuse = parg(igdRfuse)

      tskin = parg(igtskin)

      wfblim = max( min( wfb , Rfuse ) , 0.0 )
      thetafb = asin(wfblim/Rfuse)
      hfb = sqrt(Rfuse**2 - wfb**2)
      sint = wfb/Rfuse
      cost = hfb/Rfuse
      sin2t = 2.0*sint*cost
      Afuse = (pi + 2.0*thetafb + sin2t)*Rfuse**2 + 2.0*Rfuse*dRfuse
      anose = parg(iganose)
      btail = parg(igbtail)

      Rcyl = sqrt(Afuse/pi)

c---- fuselage structure and cabin parameters
      Wcabin = parg(igWpay)
     &       + parg(igWpay)*parg(igfpadd)
     &       + parg(igWpay)*parg(igfseat)
     &       + parg(igWwindow)
     &       + parg(igWinsul)
     &       + parg(igWfloor)

      Wshell = parg(igWshell)
      Whbend = parg(igWhbend)
      Wvbend = parg(igWvbend)
      Wcone  = parg(igWcone)

      EIhshell = parg(igEIhshell)
      EIhbend  = parg(igEIhbend )

      EIvshell = parg(igEIvshell)
      EIvbend  = parg(igEIvbend )

      GJshell = parg(igGJshell)
      GJcone  = parg(igGJcone )


      ni = 31
      ispace = 1

c---- set body geometry points
      do i = 1, ni
        fraci = float(i-1) / float(ni-1)
        if(ispace.eq.0) then
         frac = fraci
        else
         frac = 0.5*(1.0 - cos(pi*fraci))
        endif

        x = xnose*(1.0-frac) + xend*frac
        y = 0.

        if(x .lt. xupsweep) then
         f = 1.0 - (x-xnose)/(xupsweep-xnose)
         z = dupsweep * f**2
        else
         z = 0.
        endif

        if    (i.eq.1 .or. i.eq.ni) then
         rad = 0.

        elseif(x .lt. xblend1) then
         f = 1.0 - (x-xnose)/(xblend1-xnose)
         rad = Rcyl*(1.0 - f**anose)**(1.0/anose)

        elseif(x .lt. xblend2) then
         rad = Rcyl

        else
         f = (x-xblend2)/(xend-xblend2)
         rad = Rcyl*(1.0 - f**btail)

        endif

        IB = i
        t = x

        QB(IB,JXA,IS) = x
        TB(IB,JXA,IS) = t

        QB(IB,JYA,IS) = y
        TB(IB,JYA,IS) = t

        QB(IB,JZA,IS) = z
        TB(IB,JZA,IS) = t

        QB(IB,JRAD,IS) = rad
        TB(IB,JRAD,IS) = t

        QB(IB,JCSH,IS) = rad + tskin
        TB(IB,JCSH,IS) = t

        QB(IB,JNSH,IS) = rad + tskin
        TB(IB,JNSH,IS) = t

        QB(IB,JASH,IS) = Afuse*tskin
        TB(IB,JASH,IS) = t
      enddo

      NB(JXA,IS) = ni
      NB(JYA,IS) = ni
      NB(JZA,IS) = ni
      NB(JRAD,IS) = ni
      NB(JCSH,IS) = ni
      NB(JNSH,IS) = ni
      NB(JASH,IS) = ni

      LQBDEF(JXA,IS) = .TRUE.
      LQBDEF(JYA,IS) = .TRUE.
      LQBDEF(JZA,IS) = .TRUE.
      LQBDEF(JRAD,IS) = .TRUE.
      LQBDEF(JCSH,IS) = .TRUE.
      LQBDEF(JNSH,IS) = .TRUE.
      LQBDEF(JASH,IS) = .TRUE.


c---- fuselage shell structure and weights
      ni = 8
      do i = 1, 8
        mg1 = 0.
        mgcc1 = 0.
        mgnn1 = 0.

        mg2 = 0.
        mgcc2 = 0.
        mgnn2 = 0.

        EIcc = 0.
        EInn = 0.
        GJ = 0.

        if    (i.eq.1) then
          x = xnose

          EIcc = 0.1*EIhshell
          EInn = 0.1*EIvshell
          GJ   = 0.1*GJshell

        elseif(i.eq.2) then
          x = xshell1

          mg1 = Wshell/(xshell2-xnose)
          mgcc1 = mg1 * 0.5*(Rfuse+dRfuse)**2
          mgnn1 = mg1 * 0.5*(Rfuse+wfb)**2

          EIcc = EIhshell
          EInn = EIvshell
          GJ   = GJshell

        elseif(i.eq.3) then
          x = xshell1

          mg1 = Wshell/(xshell2-xnose)
          mgcc1 = mg1 * 0.5*(Rfuse+dRfuse)**2
          mgnn1 = mg1 * 0.5*(Rfuse+wfb)**2

          EIcc = EIhshell
          EInn = EIvshell
          GJ   = GJshell

          mg2 = Wcabin/(xshell2-xshell1)
          mgcc2 = mg2 * 0.25*(Rfuse+dRfuse)**2
          mgnn2 = mg2 * 0.50*(Rfuse+wfb)**2

        elseif(i.eq.4) then
          x = xwbox

          mg1 = Wshell/(xshell2-xnose)
     &        + Whbend*2.0/(xshell2-xshell1)
          mgcc1 = mg1 * 0.5*(Rfuse+dRfuse)**2
          mgnn1 = mg1 * 0.5*(Rfuse+wfb)**2

          EIcc = EIhshell + EIhbend
          EInn = EIvshell
          GJ   = GJshell

          mg2 = Wcabin/(xshell2-xshell1)
          mgcc2 = mg2 * 0.25*(Rfuse+dRfuse)**2
          mgnn2 = mg2 * 0.50*(Rfuse+wfb)**2

        elseif(i.eq.5) then
          x = xwbox

          mg1 = Wshell/(xshell2-xnose)
     &        + Whbend*2.0/(xshell2-xshell1)
     &        + Wvbend*2.0/(xshell2-xwbox)
          mgcc1 = mg1 * 0.5*(Rfuse+dRfuse)**2
          mgnn1 = mg1 * 0.5*(Rfuse+wfb)**2

          EIcc = EIhshell + EIhbend
          EInn = EIvshell + EIvbend
          GJ   = GJshell

          mg2 = Wcabin/(xshell2-xshell1)
          mgcc2 = mg2 * 0.25*(Rfuse+dRfuse)**2
          mgnn2 = mg2 * 0.50*(Rfuse+wfb)**2

        elseif(i.eq.6) then
          x = xshell2

          mg1 = Wshell/(xshell2-xnose)
          mgcc1 = mg1 * 0.5*(Rfuse+dRfuse)**2
          mgnn1 = mg1 * 0.5*(Rfuse+wfb)**2

          EIcc = EIhshell
          EInn = EIvshell
          GJ   = GJshell

          mg2 = Wcabin/(xshell2-xshell1)
          mgcc2 = mg2 * 0.25*(Rfuse+dRfuse)**2
          mgnn2 = mg2 * 0.50*(Rfuse+wfb)**2

        elseif(i.eq.7) then
          x = xshell2

          mg1 = Wcone/(xend-xshell2)
          mgcc1 = mg1 * 0.5*(Rfuse+dRfuse)**2
          mgnn1 = mg1 * 0.5*(Rfuse+wfb)**2

          EIcc = EIhshell
          EInn = EIvshell
          GJ   = GJcone

        elseif(i.eq.8) then
          x = xend

          EIcc = 0.2*EIhshell
          EInn = 0.2*EIvshell
          GJ   = 0.5*GJcone

        endif

        t = x

        IB = i

        QB(IB,JMG1 ,IS) = mg1
        TB(IB,JMG1 ,IS) = t
        
        QB(IB,JMCC1,IS) = mgcc1
        TB(IB,JMCC1,IS) = t

        QB(IB,JMNN1,IS) = mgnn1
        TB(IB,JMNN1,IS) = t


        QB(IB,JMG2 ,IS) = mg2
        TB(IB,JMG2 ,IS) = t
        
        QB(IB,JMNN2,IS) = mgnn2
        TB(IB,JMNN2,IS) = t

        QB(IB,JMCC2,IS) = mgcc2
        TB(IB,JMCC2,IS) = t


        QB(IB,JECC,IS) = EIcc
        TB(IB,JECC,IS) = t

        QB(IB,JENN,IS) = EInn
        TB(IB,JENN,IS) = t

        QB(IB,JGJ ,IS) = GJ
        TB(IB,JGJ ,IS) = t

      enddo


      NB(JMG1 ,IS) = ni
      NB(JMCC1,IS) = ni
      NB(JMNN1,IS) = ni
      NB(JMG2 ,IS) = ni
      NB(JMCC2,IS) = ni
      NB(JMNN2,IS) = ni
      NB(JECC,IS) = ni
      NB(JENN,IS) = ni
      NB(JGJ ,IS) = ni

      LQBDEF(JMG1 ,IS) = .TRUE.
c     LQBDEF(JMCC1,IS) = .TRUE.
c     LQBDEF(JMNN1,IS) = .TRUE.
      LQBDEF(JMG2 ,IS) = .TRUE.
      LQBDEF(JMCC2,IS) = .TRUE.
      LQBDEF(JMNN2,IS) = .TRUE.
      LQBDEF(JECC,IS) = .TRUE.
      LQBDEF(JENN,IS) = .TRUE.
      LQBDEF(JGJ ,IS) = .TRUE.

c---- fuselage aero
      ni = 21
      do i = 1, ni
        frac = float(i-1)/float(ni-1)
        x  = xnose*(1.0-frac ) + xend*frac

        Cdiss = 0.
        ue = 0.
        do ibl = 1, iblte-1
          if(x .ge. xbl(ibl) .and.
     &       x .le. xbl(ibl+1)) then
            fi = (x-xbl(ibl))/(xbl(ibl+1)-xbl(ibl))
            Cdissp = 2.0*cdbl(ibl+1)*thbl(ibl+1)/tsbl(ibl+1)
            if(tsbl(ibl) .eq. 0.0) then
             Cdisso = Cdissp
            else
             Cdisso = 2.0*cdbl(ibl)*thbl(ibl)/tsbl(ibl)
            endif
            Cdiss = Cdisso*(1.0-fi) + Cdissp*fi
          endif
          ue = uebl(ibl)*(1.0-fi) + uebl(ibl+1)*fi
        enddo

        IB = i

        t = x

        QB(IB,JCDF,IS) = 2.0*Cdiss*ue**3
        TB(IB,JCDF,IS) = t

        QB(IB,JCDP,IS) = 0.05
        TB(IB,JCDP,IS) = t
      enddo

      NB(JCDF,IS) = ni
      NB(JCDP,IS) = ni

      LQBDEF(JCDF,IS) = .TRUE.
      LQBDEF(JCDP,IS) = .TRUE.

c================================================================
c---- wing
      NBEAM = NBEAM + 1

      IS = NBEAM

      KBNUM(IS) = IS
      IBEAM(IS) = IS
      ISwing = IS

      BNAME(IS) = 'Wing'


c---------------------------------------------------------
      cosL = cos(parg(igsweep)*pi/180.0)
      tanL = tan(parg(igsweep)*pi/180.0)

      if    (iwplan.eq.0) then
       wdihed = 2.0
      elseif(iwplan.eq.1) then
       wdihed = 2.5
      else
       wdihed = 0.0
      endif

      tanD = tan(wdihed*pi/180.0)

      co = parg(igco)
      cs = parg(igco)*parg(iglambdas)
      ct = parg(igco)*parg(iglambdat)
      bo = parg(igbo)
      bs = parg(igbs)
      b  = parg(igb)

      yo = 0.5*bo
      ys = 0.5*bs
      yt = 0.5*b


      etao = bo/b
      etas = bs/b
      etat = 1.0

      dtwdL = -1.2
      etac = 0.5

      twisto = (1.0 + dtwdL*tanL*(etao-etac)) * pi/180.0
      twists = (1.0 + dtwdL*tanL*(etas-etac)) * pi/180.0
      twistt = (1.0 + dtwdL*tanL*(etat-etac)) * pi/180.0

c     if(iwplan.eq.2) then
c      twistt = 0.5 * pi/180.0
c     endif

      alphao = 1.0 * pi/180.0
      alphas = 1.0 * pi/180.0
      alphat = 1.0 * pi/180.0

      dcldao = 2.0*pi * 1.05
      dcldas = 2.0*pi * 1.05
      dcldat = 2.0*pi * 1.05

      cmo = para(iacmpo)
      cms = para(iacmps)
      cmt = para(iacmpt)

      cdfo = 0.006
      cdpo = 0.003

      cdfs = 0.006
      cdps = 0.003

      cdft = 0.006
      cdpt = 0.003


      rh   = parg(igrh)
      wbox = parg(igwbox)

      hboxo  = parg(ighboxo)
      tbcapo = parg(igtbcapo)
      tbwebo = parg(igtbwebo)
      Abcapo = 2.0*tbcapo*wbox
      Abwebo = 2.0*tbwebo*rh*hboxo
      havgo  = hboxo * (1.0 - (1.0-rh)/3.0)
      Abfuelo = (wbox-2.0*tbwebo)*(havgo-2.0*tbcapo)

      hboxs  = parg(ighboxs)
      tbcaps = parg(igtbcaps)
      tbwebs = parg(igtbwebs)
      Abcaps = 2.0*tbcaps*wbox
      Abwebs = 2.0*tbwebs*rh*hboxs
      havgs  = hboxs * (1.0 - (1.0-rh)/3.0)
      Abfuels = (wbox-2.0*tbwebs)*(havgs-2.0*tbcaps)

      rhocap  = parg(igrhocap)
      rhoweb  = parg(igrhoweb)
      rhofuel = parg(igrhofuel)

      rfmax = parg(igWfuel) / parg(igWfmax)

      Xax = parg(igXaxis)

      Xbox0 = -0.5*wbox
      Xbox1 =  0.5*wbox
      XLE   =    -Xax
      XTE   = 1.0-Xax

      zwing = parg(igzwing)

      
      mgobox = 2.0 * rhocap*gee * tbcapo*wbox     * (co*cosL)**2
     &       + 2.0 * rhoweb*gee * tbwebo*hboxo*rh * (co*cosL)**2
      mgoflap = mgobox*parg(igfflap)
      mgoslat = mgobox*parg(igfslat)
      mgoaile = mgobox*parg(igfaile)
      mgolete = mgobox*parg(igflete)
      mgoribs = mgobox*parg(igfribs)
      mgospoi = mgobox*parg(igfspoi)

      mgsbox = 2.0 * rhocap*gee * tbcaps*wbox     * (cs*cosL)**2
     &       + 2.0 * rhoweb*gee * tbwebs*hboxs*rh * (cs*cosL)**2
      mgsflap = mgsbox*parg(igfflap)
      mgsslat = mgsbox*parg(igfslat)
      mgsaile = mgsbox*parg(igfaile)
      mgslete = mgsbox*parg(igflete)
      mgsribs = mgsbox*parg(igfribs)
      mgsspoi = mgsbox*parg(igfspoi)

      mgo = mgobox
     &    + mgoflap
     &    + mgoslat
     &    + mgoaile
     &    + mgolete
     &    + mgoribs
     &    + mgospoi
      mgs = mgsbox
     &    + mgsflap
     &    + mgsslat
     &    + mgsaile
     &    + mgslete
     &    + mgsribs
     &    + mgsspoi

      dXmgo  = mgobox  *  0.0
     &       + mgoflap * (Xbox1 + 0.20*(XTE-Xbox1))
     &       + mgoslat * (Xbox0 + 0.50*(XLE-Xbox0))
     &       + mgoaile * (Xbox1 + 0.40*(XTE-Xbox1))
     &       + mgolete *  Xbox0
     &       + mgoribs *  0.0
     &       + mgospoi * (Xbox1 + 0.05*(XTE-Xbox1))
      dXmgs  = mgsbox  *  0.0
     &       + mgsflap * (Xbox1 + 0.20*(XTE-Xbox1))
     &       + mgsslat * (Xbox0 + 0.50*(XLE-Xbox0))
     &       + mgsaile * (Xbox1 + 0.40*(XTE-Xbox1))
     &       + mgslete *  Xbox0
     &       + mgsribs *  0.0
     &       + mgsspoi * (Xbox1 + 0.05*(XTE-Xbox1))

      dX2mgo = mgobox  *  wbox**2 / 12.0
     &       + mgoflap * (Xbox1 + 0.20*(XTE-Xbox1))**2
     &       + mgoslat * (Xbox0 + 0.50*(XLE-Xbox0))**2
     &       + mgoaile * (Xbox1 + 0.40*(XTE-Xbox1))**2
     &       + mgolete *  Xbox0**2
     &       + mgoribs *  wbox**2 / 12.0
     &       + mgospoi * (Xbox1 + 0.05*(XTE-Xbox1))**2
      dX2mgs = mgsbox  *  wbox**2 / 12.0
     &       + mgsflap * (Xbox1 + 0.20*(XTE-Xbox1))**2
     &       + mgsslat * (Xbox0 + 0.50*(XLE-Xbox0))**2
     &       + mgsaile * (Xbox1 + 0.40*(XTE-Xbox1))**2
     &       + mgslete *  Xbox0**2
     &       + mgsribs *  wbox**2 / 12.0
     &       + mgsspoi * (Xbox1 + 0.05*(XTE-Xbox1))**2

      Ccgo = (dXmgo/mgo)*co*cosL ! + 0.2*co*cosL
      Ccgs = (dXmgs/mgs)*cs*cosL ! + 0.2*cs*cosL

      mgnno = dX2mgo*(co*cosL)**2 - mgo*Ccgo**2
      mgnns = dX2mgs*(cs*cosL)**2 - mgs*Ccgs**2

      mgofuel = Abfuelo*rhofuel*gee * rfmax * (co*cosL)**2
      mgsfuel = Abfuels*rhofuel*gee * rfmax * (cs*cosL)**2

      mgnnofuel = mgofuel * (wbox*co*cosL)**2 / 12.0
      mgnnsfuel = mgsfuel * (wbox*cs*cosL)**2 / 12.0

      Csho = 0.5*wbox*co*cosL
      Cshs = 0.5*wbox*cs*cosL

      Nsho = 0.5*hboxo*co*cosL
      Nshs = 0.5*hboxs*cs*cosL

      Atsho = (wbox-tbwebo)*(havgo-tbcapo)*tbcapo*(co*cosL)**3
      Atshs = (wbox-tbwebs)*(havgs-tbcaps)*tbcaps*(cs*cosL)**3


      EIcco = parg(igEIco)
      EInno = parg(igEIno)
      GJo   = parg(igGJo)
      EAo   = parg(igEcap) * 2.0*tbcapo*wbox * (co*cosL)**2

      EIccs = parg(igEIcs)
      EInns = parg(igEIns)
      GJs   = parg(igGJs)
      EAs   = parg(igEcap) * 2.0*tbcaps*wbox * (cs*cosL)**2



!!      GJo = GJo * 0.1
!!      GJs = GJs * 0.1


      i1 = 1
      i2 = 2
      do i = i1, i2
        frac = float(i-i1)/float(i2-i1)

        y = yo * frac
        c = co

        x = xwbox
        z = zwing

        t = y

        twist = twisto
        alpha = alphao
        dclda = dcldao
        cm    = cmo
        cdf   = cdfo
        cdp   = cdpo
        mg    = mgo
        mgnn  = mgnno
        Ccg   = Ccgo
        mgf   = mgofuel
        mgnnf = mgnnofuel
        Ccgf  = 0.
        EIcc  = EIcco
        EInn  = EInno
        GJ    = GJo
        EA    = EAo * EAfac
        Csh   = Csho
        Nsh   = Nsho
        Atsh  = Atsho
        EIcs  = 0.
        EIsn  = 0.

        if(ifwcen .eq. 0) then
         mgf   = 0.
         mgnnf = 0.
        endif

        IB = i

        QB(IB,JXAX,IS) = Xax
        QB(IB,JXA,IS) = x
        QB(IB,JYA,IS) = y
        QB(IB,JZA,IS) = z
        QB(IB,JCH,IS) = c
        QB(IB,JTW,IS) = twist
        QB(IB,JAL,IS) = alpha
        QB(IB,JDCL,IS) = dclda
        QB(IB,JCM ,IS) = cm
        QB(IB,JCDF,IS) = cdf
        QB(IB,JCDP,IS) = cdp
        QB(IB,JMG1,IS) = mg
        QB(IB,JMNN1,IS) = mgnn
        QB(IB,JCCG1,IS) = Ccg
        QB(IB,JMG2 ,IS) = mgf
        QB(IB,JMNN2,IS) = mgnnf
        QB(IB,JCCG2,IS) = Ccgf
        QB(IB,JECC,IS) = EIcc
        QB(IB,JENN,IS) = EInn
        QB(IB,JGJ ,IS) = GJ
        QB(IB,JECS,IS) = EIcs
        QB(IB,JESN,IS) = EIsn
        QB(IB,JEA ,IS) = EA
        QB(IB,JCSH,IS) = Csh
        QB(IB,JNSH,IS) = Nsh
        QB(IB,JASH,IS) = Atsh
        DO J = 1, JBTOT
          TB(IB,J,IS) = t
        ENDDO
      enddo
      ni = i2


      i1 = ni+1
      i2 = ni+4
      do i = i1, i2
        frac = float(i-i1)/float(i2-i1)

        y = yo*(1.0-frac) + ys*frac
        c = co*(1.0-frac) + cs*frac
        cnorm = c * cosL

        x = xwbox + (y-yo)*tanL
        z = zwing + (y-yo)*tanD

        t = y

        twist = twisto*(1.0-frac) + twists*frac
        alpha = alphao*(1.0-frac) + alphas*frac
        dclda = dcldao*(1.0-frac) + dcldas*frac

        cm    = cmo *(1.0-frac) + cms *frac
        cdf   = cdfo*(1.0-frac) + cdfs*frac
        cdp   = cdpo*(1.0-frac) + cdps*frac

        mg    = mgo  *(1.0-frac) + mgs  *frac
        mgnn  = mgnno*(1.0-frac) + mgnns*frac
        Ccg   = Ccgo *(1.0-frac) + Ccgs *frac

        mgf   = mgofuel  *(1.0-frac) + mgsfuel*frac
        mgnnf = mgnnofuel*(1.0-frac) + mgsfuel*frac
        Ccgf  = 0.

        EIcc  = EIcco*(1.0-frac) + EIccs*frac
        EInn  = EInno*(1.0-frac) + EInns*frac
        GJ    = GJo  *(1.0-frac) + GJs  *frac
        EA    = EAo  *(1.0-frac) + EAs  *frac

        Csh   = Csho *(1.0-frac) + Csh *frac
        Nsh   = Nsho *(1.0-frac) + Nsh *frac
        Atsh  = Atsho*(1.0-frac) + Atsh*frac


        IB = i

        QB(IB,JXAX,IS) = Xax
        QB(IB,JXA,IS) = x
        QB(IB,JYA,IS) = y
        QB(IB,JZA,IS) = z
        QB(IB,JCH,IS) = cnorm
        QB(IB,JTW,IS) = twist
        QB(IB,JAL,IS) = alpha
        QB(IB,JDCL,IS) = dclda
        QB(IB,JCM ,IS) = cm
        QB(IB,JCDF,IS) = cdf
        QB(IB,JCDP,IS) = cdp
        QB(IB,JMG1,IS) = mg
        QB(IB,JMNN1,IS) = mgnn
        QB(IB,JCCG1,IS) = Ccg
        QB(IB,JMG2 ,IS) = mgf
        QB(IB,JMNN2,IS) = mgnnf
        QB(IB,JCCG2,IS) = Ccgf
        QB(IB,JECC,IS) = EIcc
        QB(IB,JENN,IS) = EInn
        QB(IB,JGJ ,IS) = GJ
        QB(IB,JEA ,IS) = EA
        QB(IB,JCSH,IS) = Csh
        QB(IB,JNSH,IS) = Nsh
        QB(IB,JASH,IS) = Atsh
        DO J = 1, JBTOT
          TB(IB,J,IS) = t
        ENDDO
      enddo
      ni = i2


      i1 = ni+1
      i2 = ni+5
      do i = i1, i2
        frac = float(i-i1)/float(i2-i1)

        y = ys*(1.0-frac) + yt*frac
        c = cs*(1.0-frac) + ct*frac
        cnorm = c * cosL

        fc = c/cs

        x = xwbox + (y-yo)*tanL
        z = zwing + (y-yo)*tanD

        t = y

        twist = twists*(1.0-frac) + twistt*frac
        alpha = alphas*(1.0-frac) + alphat*frac
        dclda = dcldas*(1.0-frac) + dcldat*frac

        cm    = cms *(1.0-frac) + cms *frac
        cdf   = cdfs*(1.0-frac) + cdfs*frac
        cdp   = cdps*(1.0-frac) + cdps*frac

        mg    = mgs  *fc**2
        mgnn  = mgnns*fc**4
        Ccg   = Ccgs *fc

        mgf   = mgsfuel  *fc**2
        mgnnf = mgnnsfuel*fc**4
        Ccgf  = 0.

        EIcc  = EIccs*fc**4
        EInn  = EInns*fc**4
        GJ    = GJs  *fc**4
        EA    = EAs  *fc**2

        Csh   = Cshs *fc
        Nsh   = Nshs *fc
        Atsh  = Atshs*fc**3


        IB = i

        QB(IB,JXAX,IS) = Xax
        QB(IB,JXA,IS) = x
        QB(IB,JYA,IS) = y
        QB(IB,JZA,IS) = z
        QB(IB,JCH,IS) = cnorm
        QB(IB,JTW,IS) = twist
        QB(IB,JAL,IS) = alpha
        QB(IB,JDCL,IS) = dclda
        QB(IB,JCM ,IS) = cm
        QB(IB,JCDF,IS) = cdf
        QB(IB,JCDP,IS) = cdp
        QB(IB,JMG1,IS) = mg
        QB(IB,JMNN1,IS) = mgnn
        QB(IB,JCCG1,IS) = Ccg
        QB(IB,JMG2 ,IS) = mgf
        QB(IB,JMNN2,IS) = mgnnf
        QB(IB,JCCG2,IS) = Ccgf
        QB(IB,JECC,IS) = EIcc
        QB(IB,JENN,IS) = EInn
        QB(IB,JGJ ,IS) = GJ
        QB(IB,JEA ,IS) = EA
        QB(IB,JCSH,IS) = Csh
        QB(IB,JNSH,IS) = Nsh
        QB(IB,JASH,IS) = Atsh
        DO J = 1, JBTOT
          TB(IB,J,IS) = t
        ENDDO
      enddo
      ni = i2

      DO J = 1, JBTOT
        NB(J,IS) = ni
      ENDDO

      LQBDEF(JXA,IS) = .TRUE.
      LQBDEF(JYA,IS) = .TRUE.
      LQBDEF(JZA,IS) = .TRUE.
      LQBDEF(JTW,IS) = .TRUE.
      LQBDEF(JCH,IS) = .TRUE.
      LQBDEF(JAL,IS) = .TRUE.
      LQBDEF(JXAX,IS) = .TRUE.
      LQBDEF(JDCL,IS) = .TRUE.
      LQBDEF(JCM ,IS) = .TRUE.
      LQBDEF(JCDF,IS) = .TRUE.
      LQBDEF(JCDP,IS) = .TRUE.
      LQBDEF(JMG1,IS) = .TRUE.
      LQBDEF(JMNN1,IS) = .TRUE.
      LQBDEF(JCCG1,IS) = .TRUE.
      LQBDEF(JMG2 ,IS) = .TRUE.
      LQBDEF(JMNN2,IS) = .TRUE.
      LQBDEF(JCCG2,IS) = .TRUE.
      LQBDEF(JECC,IS) = .TRUE.
      LQBDEF(JENN,IS) = .TRUE.
      LQBDEF(JGJ ,IS) = .TRUE.
      LQBDEF(JEA ,IS) = .TRUE.
      LQBDEF(JCSH,IS) = .TRUE.
      LQBDEF(JNSH,IS) = .TRUE.
      LQBDEF(JASH,IS) = .TRUE.

c================================================================
c---- horizontal tail
      NBEAM = NBEAM + 1

      IS = NBEAM

      KBNUM(IS) = IS
      IBEAM(IS) = IS
      IShtail = IS

      BNAME(IS) = 'Horizontal Tail'


c---------------------------------------------------------
      cosLh = cos(parg(igsweeph)*pi/180.0)
      tanLh = tan(parg(igsweeph)*pi/180.0)

      hdihed = 0.0
      tanDh = tan(hdihed*pi/180.0)

      coh = parg(igcoh)
      cth = parg(igcoh)*parg(iglambdah)
      boh = parg(igboh)
      bh  = parg(igbh)

      yoh = 0.5*boh
      yth = 0.5*bh

      twisth =  0.0 * pi/180.0
      alphah = -1.0 * pi/180.0
      dcldah = 2.0*pi * 1.05
      dcldfh =  0.08
      dcmdfh = -0.012

      cmh  = 0.
      cdfh = para(iacdft) * para(iafexcdt)
      cdph = para(iacdpt) * para(iafexcdt)

      rhh   = parg(igrhh)
      wboxh = parg(igwboxh)

      hboxh  = parg(ighboxh)
      tbcaph = parg(igtbcaph)
      tbwebh = parg(igtbwebh)
      Abcaph = 2.0*tbcaph*wboxh
      Abwebh = 2.0*tbwebh*rhh*hboxh
      havgh  = hboxh * (1.0 - (1.0-rhh)/3.0)

      rhocaph = parg(igrhocap)
      rhowebh = parg(igrhoweb)

      Xaxh = parg(igXaxis)

      Xbox0h = -0.5*wboxh
      Xbox1h =  0.5*wboxh
      XLEh   =    -Xaxh
      XTEh   = 1.0-Xaxh

      xhbox  = parg(igxhbox)
      zhtail = parg(igzhtail)

      if(Lptail) then
       zhtail = parg(igbv)
      endif
      
      mghbox = 2.0 * rhocaph*gee * tbcaph*wboxh    * (coh*cosLh)**2
     &       + 2.0 * rhowebh*gee * tbwebh*hboxh*rhh* (coh*cosLh)**2
      mghadd = mghbox*parg(igfhadd)

      mgh = mghbox
     &    + mghadd
      dXmgh  = mghbox *  0.0
     &       + mghadd *  0.0
      dX2mgh = mghbox *  wboxh**2 / 12.0
     &       + mghadd * (wboxh*0.5)**2

      Ccgh = (dXmgh/mgh)*coh*cosLh

      mgnnh = dX2mgh*(coh*cosLh)**2 - mgh*Ccgh**2

      Cshh = 0.5*wboxh*coh*cosLh
      Nshh = 0.5*hboxh*coh*cosLh
      Atshh = (wboxh-tbwebh)*(havgh-tbcaph)*tbcaph*(coh*cosLh)**3

      EIcch = parg(igEIch)
      EInnh = parg(igEInh)
      GJh   = parg(igGJh)
      EAh   = parg(igEcap) * 2.0*tbcaph*wboxh * (coh*cosLh)**2

      if(yoh .eq. 0.0) then
       ni = 0
      else
       i1 = 1
       i2 = 2
       do i = i1, i2
         frac = float(i-i1)/float(i2-i1)

         y = yoh * frac
         c = coh

         x = xhbox
         z = zhtail

         t = y

         twist = twisth
         alpha = alphah
         dclda = dcldah
         dcldf = dcldfh
         dcmdf = dcmdfh

         cm    = cmh
         cdf   = cdfh
         cdp   = cdph

         mg    = mgh
         mgnn  = mgnnh

         Ccg   = Ccgh

         EIcc  = EIcch
         EInn  = EInnh
         GJ    = GJh
         EA    = EAh   * EAfac

         Csh   = Cshh
         Nsh   = Nshh
         Atsh  = Atshh

         IB = i

         QB(IB,JXAX,IS) = Xaxh
         QB(IB,JXA,IS) = x
         QB(IB,JYA,IS) = y
         QB(IB,JZA,IS) = z
         QB(IB,JCH,IS) = c
         QB(IB,JTW,IS) = twist
         QB(IB,JAL,IS) = alpha

         QB(IB,JDCL,IS) = dclda
         QB(IB,JCLF2,IS) = dcldf
         QB(IB,JCMF2,IS) = dcmdf

         QB(IB,JCM ,IS) = cm
         QB(IB,JCDF,IS) = cdf
         QB(IB,JCDP,IS) = cdp

         QB(IB,JMG1,IS) = mg
         QB(IB,JMNN1,IS) = mgnn
         QB(IB,JCCG1,IS) = Ccg
         QB(IB,JECC,IS) = EIcc
         QB(IB,JENN,IS) = EInn
         QB(IB,JGJ ,IS) = GJ
         QB(IB,JEA ,IS) = EA

         QB(IB,JCSH,IS) = Csh
         QB(IB,JNSH,IS) = Nsh
         QB(IB,JASH,IS) = Atsh
         DO J = 1, JBTOT
           TB(IB,J,IS) = t
         ENDDO
       enddo
       ni = i2
      endif


      i1 = ni+1
      i2 = ni+5
      do i = i1, i2
        frac = float(i-i1)/float(i2-i1)

        y = yoh*(1.0-frac) + yth*frac
        c = coh*(1.0-frac) + cth*frac
        cnorm = c * cosLh

        fc = c/coh

        x = xhbox  + (y-yoh)*tanLh
        z = zhtail + (y-yoh)*tanDh

        t = y

        twist = twisth
        alpha = alphah
        dclda = dcldah
        dcldf = dcldfh
        dcmdf = dcmdfh

        cm    = cmh 
        cdf   = cdfh
        cdp   = cdph

        mg    = mgh  *fc**2
        mgnn  = mgnnh*fc**4
        Ccg   = Ccgh *fc

        EIcc  = EIcch*fc**4
        EInn  = EInnh*fc**4
        GJ    = GJh  *fc**4
        EA    = EAh  *fc**2

        Csh   = Cshh *fc
        Nsh   = Nshh *fc
        Atsh  = Atshh*fc**3


        IB = i

        QB(IB,JXAX,IS) = Xaxh
        QB(IB,JXA,IS) = x
        QB(IB,JYA,IS) = y
        QB(IB,JZA,IS) = z
        QB(IB,JCH,IS) = cnorm
        QB(IB,JTW,IS) = twist
        QB(IB,JAL,IS) = alpha

        QB(IB,JDCL,IS) = dclda
        QB(IB,JCLF2,IS) = dcldf
        QB(IB,JCMF2,IS) = dcmdf
        QB(IB,JCM ,IS) = cm
        QB(IB,JCDF,IS) = cdf
        QB(IB,JCDP,IS) = cdp

        QB(IB,JMG1,IS) = mg
        QB(IB,JMNN1,IS) = mgnn
        QB(IB,JCCG1,IS) = Ccg
        QB(IB,JECC,IS) = EIcc
        QB(IB,JENN,IS) = EInn
        QB(IB,JGJ ,IS) = GJ
        QB(IB,JEA ,IS) = EA

        QB(IB,JCSH,IS) = Csh
        QB(IB,JNSH,IS) = Nsh
        QB(IB,JASH,IS) = Atsh
        DO J = 1, JBTOT
          TB(IB,J,IS) = t
        ENDDO
      enddo
      ni = i2

      DO J = 1, JBTOT
        NB(J,IS) = ni
      ENDDO

      LQBDEF(JXA,IS) = .TRUE.
      LQBDEF(JYA,IS) = .TRUE.
      LQBDEF(JZA,IS) = .TRUE.
      LQBDEF(JTW,IS) = .TRUE.
      LQBDEF(JCH,IS) = .TRUE.
      LQBDEF(JAL,IS) = .TRUE.
      LQBDEF(JXAX,IS) = .TRUE.
      LQBDEF(JDCL,IS) = .TRUE.
      LQBDEF(JCLF2,IS)= .TRUE.
      LQBDEF(JCMF2,IS)= .TRUE.
      LQBDEF(JCM ,IS) = .TRUE.
      LQBDEF(JCDF,IS) = .TRUE.
      LQBDEF(JCDP,IS) = .TRUE.
      LQBDEF(JMG1,IS) = .TRUE.
      LQBDEF(JMNN1,IS) = .TRUE.
      LQBDEF(JCCG1,IS) = .TRUE.
      LQBDEF(JECC,IS) = .TRUE.
      LQBDEF(JENN,IS) = .TRUE.
      LQBDEF(JGJ ,IS) = .TRUE.
      LQBDEF(JEA ,IS) = .TRUE.
      LQBDEF(JCSH,IS) = .TRUE.
      LQBDEF(JNSH,IS) = .TRUE.
      LQBDEF(JASH,IS) = .TRUE.


c================================================================
c---- vertical tail(s)
      NBEAM = NBEAM + 1

      IS = NBEAM

      KBNUM(IS) = IS
      IBEAM(IS) = IBEAM(IShtail)  ! this is aerodynamically connected to HT

      BNAME(IS) = 'Vertical Tail'

c---------------------------------------------------------
      cosLv = cos(parg(igsweepv)*pi/180.0)
      tanLv = tan(parg(igsweepv)*pi/180.0)

      cov = parg(igcov)
      ctv = parg(igcov)*parg(iglambdav)

      twistv = 0. * pi/180.0
      alphav = 0. * pi/180.0
      dcldav = 2.0*pi * 1.05

      cmv  = 0.
      cdfv = para(iacdft) * para(iafexcdt)
      cdpv = para(iacdpt) * para(iafexcdt)

      rhv   = parg(igrhv)
      wboxv = parg(igwboxv)

      hboxv  = parg(ighboxv)
      tbcapv = parg(igtbcapv)
      tbwebv = parg(igtbwebv)
      Abcapv = 2.0*tbcapv*wboxv
      Abwebv = 2.0*tbwebv*rhv*hboxv
      havgv  = hboxv * (1.0 - (1.0-rhv)/3.0)

      rhocapv = parg(igrhocap)
      rhowebv = parg(igrhoweb)

      Xaxv = parg(igXaxis)

      Xbox0v = -0.5*wboxv
      Xbox1v =  0.5*wboxv
      XLEv   =    -Xaxv
      XTEv   = 1.0-Xaxv

      xvbox  = parg(igxvbox)
      
      mgvbox = 2.0 * rhocapv*gee * tbcapv*wboxv    * (cov*cosLv)**2
     &       + 2.0 * rhowebv*gee * tbwebv*hboxv*rhv* (cov*cosLv)**2
      mgvadd = mgvbox*parg(igfvadd)

      mgv = mgvbox
     &    + mgvadd
      dXmgv  = mgvbox *  0.0
     &       + mgvadd *  0.0
      dX2mgv = mgvbox *  wboxv**2 / 12.0
     &       + mgvadd * (wboxv*0.5)**2

      Ccgv = (dXmgv/mgv)*cov*cosLv

      mgnnv = dX2mgv*(cov*cosLv)**2 - mgv*Ccgv**2

      Cshv = 0.5*wboxv*cov*cosLv
      Nshv = 0.5*hboxv*cov*cosLv
      Atshv = (wboxv-tbwebv)*(havgv-tbcapv)*tbcapv*(cov*cosLv)**3

      EIccv = parg(igEIcv)
      EInnv = parg(igEInv)
      GJv   = parg(igGJv)

c---- if two VT's, set horizontal connection between them
      if(Lptail) then
       i1 = 1
       i2 = 2
       t1v = t0v + yov
      else
       i1 = 1
       i2 = 0
       t1v = t0v
      endif

      do i = i1, i2
        frac = float(i-i1)/float(i2-i1)

        z = zov
        c = cov

        x = xvbox
        y = yov*frac

        t = t0v + y

        twist = twistv
        alpha = alphav
        dclda = dcldav * 0.3
        cm    = 0.
        cdf   = cdfv
        cdp   = cdpv
        mg    = 0.
        mgnn  = 0.
        Ccg   = 0.
        EIcc  = 0.
        EInn  = 0.
        GJ    = 0.
        Csh   = Csh
        Nsh   = Nshv
        Atsh  = Atshv * 10.0


        IB = i

        QB(IB,JXAX,IS) = Xaxv
        QB(IB,JXA,IS) = x
        QB(IB,JYA,IS) = y
        QB(IB,JZA,IS) = z
        QB(IB,JCH,IS) = c
        QB(IB,JTW,IS) = twist
        QB(IB,JAL,IS) = alpha
        QB(IB,JDCL,IS) = dclda
        QB(IB,JCM ,IS) = cm
        QB(IB,JCDF,IS) = cdf
        QB(IB,JCDP,IS) = cdp
        QB(IB,JMG1,IS) = mg
        QB(IB,JMNN1,IS) = mgnn
        QB(IB,JCCG1,IS) = Ccg
        QB(IB,JECC,IS) = EIcc
        QB(IB,JENN,IS) = EInn
        QB(IB,JGJ ,IS) = GJ
        QB(IB,JCSH,IS) = Csh
        QB(IB,JNSH,IS) = Nsh
        QB(IB,JASH,IS) = Atsh
        DO J = 1, JBTOT
          TB(IB,J,IS) = t
        ENDDO
      enddo
      ni = i2

      i1 = ni+1
      i2 = ni+5
      do i = i1, i2
        frac = float(i-i1)/float(i2-i1)

        z = zov + ztv*frac
        c = cov*(1.0-frac) + ctv*frac
        cnorm = c * cosLv

        fc = c/cov

        x = xvbox + ztv*frac*tanLv

        y = yov*(1.0-frac) + ytv*frac

        t = t1v + ztv*frac

        twist = twistv
        alpha = alphav
        dclda = dcldav
        cm    = cmv
        cdf   = cdfv
        cdp   = cdpv
        mg    = mgv   * fc**2
        mgnn  = mgnnv * fc**4
        Ccg   = Ccgv  * fc
        EIcc  = EIccv * fc**4
        EInn  = EInnv * fc**4
        GJ    = GJv   * fc**4
        Csh   = Cshv  * fc
        Nsh   = Nshv  * fc
        Atsh  = Atshv * fc**3

        IB = i

        QB(IB,JXAX,IS) = Xaxv
        QB(IB,JXA,IS) = x
        QB(IB,JYA,IS) = y
        QB(IB,JZA,IS) = z
        QB(IB,JCH,IS) = cnorm
        QB(IB,JTW,IS) = twist
        QB(IB,JAL,IS) = alpha
        QB(IB,JDCL,IS) = dclda
        QB(IB,JCM ,IS) = cm
        QB(IB,JCDF,IS) = cdf
        QB(IB,JCDP,IS) = cdp
        QB(IB,JMG1,IS) = mg
        QB(IB,JMNN1,IS) = mgnn
        QB(IB,JCCG1,IS) = Ccg
        QB(IB,JECC,IS) = EIcc
        QB(IB,JENN,IS) = EInn
        QB(IB,JGJ ,IS) = GJ
        QB(IB,JCSH,IS) = Csh
        QB(IB,JNSH,IS) = Nsh
        QB(IB,JASH,IS) = Atsh
        DO J = 1, JBTOT
          TB(IB,J,IS) = t
        ENDDO
      enddo
      ni = i2

      DO J = 1, JBTOT
        NB(J,IS) = ni
      ENDDO

      LQBDEF(JXA,IS) = .TRUE.
      LQBDEF(JYA,IS) = .TRUE.
      LQBDEF(JZA,IS) = .TRUE.
      LQBDEF(JTW,IS) = .TRUE.
      LQBDEF(JCH,IS) = .TRUE.
      LQBDEF(JAL,IS) = .TRUE.
      LQBDEF(JXAX,IS) = .TRUE.
      LQBDEF(JDCL,IS) = .TRUE.
      LQBDEF(JCM ,IS) = .TRUE.
      LQBDEF(JCDF,IS) = .TRUE.
      LQBDEF(JCDP,IS) = .TRUE.
      LQBDEF(JMG1,IS) = .TRUE.
      LQBDEF(JMNN1,IS) = .TRUE.
      LQBDEF(JCCG1,IS) = .TRUE.
      LQBDEF(JECC,IS) = .TRUE.
      LQBDEF(JENN,IS) = .TRUE.
      LQBDEF(JGJ ,IS) = .TRUE.
      LQBDEF(JCSH,IS) = .TRUE.
      LQBDEF(JNSH,IS) = .TRUE.
      LQBDEF(JASH,IS) = .TRUE.


c================================================================
      if(iwplan .eq. 2) then
c---- strut
      NBEAM = NBEAM + 1

      IS = NBEAM

      KBNUM(IS) = IS
      IBEAM(IS) = IBEAM(ISwing)  ! this is aerodynamically connected to Wing
      BNAME(IS) = 'Strut'

      bo = parg(igbo)
      bs = parg(igbs)
      b  = parg(igb)

      yo = 0.5*bo
      ys = 0.5*bs

      dzs = -0.1*parg(igco)*parg(iglambdas)

      xo = xwbox
      xs = xwbox + (ys-yo)*tanL

      zo = zwing - parg(igzs)
      zs = zwing + (ys-yo)*tanD + dzs
      
      cosLs = parg(igcosLs)

      twist = 0.

      rhostrut = parg(igrhostrut)
      Estrut   = parg(igEcap)
      Gstrut   = 0.5*Estrut/(1.0+0.3)

c      cstrut = sqrt(0.5*Astrut/(tohstrut*hstrut))

      Astrut = parg(igAstrut)   ! A
      hstrut = parg(ighstrut)   ! h/c
      cstrut = parg(igcstrut)   ! c
      tohstrut = 0.5*Astrut/(hstrut*cstrut**2)   ! t/h
      dclda  = 2.0*pi * 0.9

      cdf = para(iacdfs)
      cdp = para(iacdps)

      Xax = 0.25
      Csh = cstrut*(1.0-Xax)
      Nsh = cstrut*hstrut*0.5

      EA   = Estrut*Astrut
      EIcc = Estrut * 0.14 * Astrut * cstrut**2 * hstrut**2
      EInn = Estrut * 0.08 * Astrut * cstrut**2 * hstrut
      GJ   = Gstrut * 0.40 * Astrut * cstrut**2 * hstrut**2

      mg   = rhostrut*gee*Astrut
      mgcc = mg * 0.07 * hstrut**2 * cstrut**2
      mgnn = mg * 0.07             * cstrut**2


      i1 = 1
      i2 = 2
      do i = i1, i2
        frac = float(i-i1)/float(i2-i1)

        x = xo      
        y = yo*frac
        z = zo     

        c = cstrut
        cnorm = c

        t = y

        IB = i

        QB(IB,JXAX,IS) = Xax
        QB(IB,JXA,IS) = x
        QB(IB,JYA,IS) = y
        QB(IB,JZA,IS) = z
        QB(IB,JCH,IS) = cnorm
        QB(IB,JTW,IS) = twist
        QB(IB,JDCL,IS) = dclda
        QB(IB,JCDF,IS) = cdf
        QB(IB,JCDP,IS) = cdp
        QB(IB,JMG1,IS) = mg
        QB(IB,JMCC1,IS) = mgcc
        QB(IB,JMNN1,IS) = mgnn
        QB(IB,JECC,IS) = EIcc
        QB(IB,JENN,IS) = EInn
        QB(IB,JGJ ,IS) = GJ
        QB(IB,JEA ,IS) = EA
        QB(IB,JCSH,IS) = Csh
        QB(IB,JNSH,IS) = Nsh
        DO J = 1, JBTOT
          TB(IB,J,IS) = t
        ENDDO
      enddo
      ni = i2

      i1 = ni + 1
      i2 = ni + 4
      do i = i1, i2
        frac = float(i-i1)/float(i2-i1)

        x = xo*(1.0-frac) + xs*frac
        y = yo*(1.0-frac) + ys*frac
        z = zo*(1.0-frac) + zs*frac

        c = cstrut
        cnorm = c * cosLs

        t = y

        IB = i

        QB(IB,JXAX,IS) = Xax
        QB(IB,JXA,IS) = x
        QB(IB,JYA,IS) = y
        QB(IB,JZA,IS) = z
        QB(IB,JCH,IS) = cnorm
        QB(IB,JTW,IS) = twist
        QB(IB,JDCL,IS) = dclda
        QB(IB,JCDF,IS) = cdf
        QB(IB,JCDP,IS) = cdp
        QB(IB,JMG1,IS) = mg
        QB(IB,JMCC1,IS) = mgcc
        QB(IB,JMNN1,IS) = mgnn
        QB(IB,JECC,IS) = EIcc
        QB(IB,JENN,IS) = EInn
        QB(IB,JGJ ,IS) = GJ
        QB(IB,JEA ,IS) = EA
        QB(IB,JCSH,IS) = Csh
        QB(IB,JNSH,IS) = Nsh
        DO J = 1, JBTOT
          TB(IB,J,IS) = t
        ENDDO
      enddo
      ni = i2

      DO J = 1, JBTOT
        NB(J,IS) = ni
      ENDDO

      LQBDEF(JXAX,IS) = .TRUE.
      LQBDEF(JXA,IS) = .TRUE.
      LQBDEF(JYA,IS) = .TRUE.
      LQBDEF(JZA,IS) = .TRUE.
      LQBDEF(JCH,IS) = .TRUE.
      LQBDEF(JTW,IS) = .TRUE.
      LQBDEF(JDCL,IS) = .TRUE.
      LQBDEF(JCDF,IS) = .TRUE.
      LQBDEF(JCDP,IS) = .TRUE.
      LQBDEF(JMG1,IS) = .TRUE.
      LQBDEF(JMNN1,IS) = .TRUE.
      LQBDEF(JECC,IS) = .TRUE.
      LQBDEF(JENN,IS) = .TRUE.
      LQBDEF(JGJ ,IS) = .TRUE.
      LQBDEF(JEA ,IS) = .TRUE.
      LQBDEF(JCSH,IS) = .TRUE.
      LQBDEF(JNSH,IS) = .TRUE.

c---- ground attach point of beam
      NGROU = NGROU + 1
      TGROU(NGROU) = yo
      KBGROU(NGROU) = IS
      KGTYPE(NGROU) = 0

      NGROU = NGROU + 1
      TGROU(NGROU) = -yo
      KBGROU(NGROU) = IS
      KGTYPE(NGROU) = 0


c---- join beam to wing
      NJOIN = NJOIN + 1
      KBJOIN(1,NJOIN) = IS
      KBJOIN(2,NJOIN) = ISwing
      TJOIN(1,NJOIN) = ys
      TJOIN(2,NJOIN) = ys
      KJTYPE(NJOIN) = 0

      NJOIN = NJOIN + 1
      KBJOIN(1,NJOIN) = IS
      KBJOIN(2,NJOIN) = ISwing
      TJOIN(1,NJOIN) = -ys
      TJOIN(2,NJOIN) = -ys
      KJTYPE(NJOIN) = 0



      endif

c---------------------------------------------------------

      call BOUTPUT(lu,IBX,JBX, LQBDEF,
     &               NBX, NBEAM, NB, QB, TB, BNAME, KBNUM,IBEAM,
     &               NJX, NJOIN, TJOIN, KBJOIN, KJTYPE,
     &               NGX, NGROU, TGROU, KBGROU, KGTYPE,
     &          KPX, NPX, NPYLO, QPYLO, KBPYLO, KPTYPE,
     &        NENGX, IENGTYP,
     &               NAJX,NANGJ, MOMJ, ANGJ, HJAX,
     &               UNITL, UNITM, UNITT, UNITF,
     &               UNCHL, UNCHM, UNCHT, UNCHF,
     &               gee, rhoSL, aSL, SREF,CREF,BREF,XYZREF,
     &               configname, LBSYMM)

      return
      end ! aswout
