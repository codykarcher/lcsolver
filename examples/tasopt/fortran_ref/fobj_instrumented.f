
      subroutine fobj(nvopt,vopt,
     &                ncon,con, fun,func, dfrel, ch,istep)
c========================================================================
c     Evaluates objective function for optimization variables vopt(.)
c     Also adds on any penalty terms from violated constraints
c========================================================================
      implicit real (a-h,l-z)
      integer nvopt, ncon, istep
      real vopt(*), con(*)
      character*(*) ch

      include 'index.inc'
      include 'constants.inc'
      include 'airf.inc'
      include 'arrs.inc'

      real penfac(iototal)

      logical Lconv

c---- put values in vopt(.) into parg,para,pare
      call voptset(nvopt,vopt)

c---- size the aircraft for the first mission
      kmdes = 1
      iairf = 1
      call wsize(pari,
     &           parg,
     &           parm(1,kmdes),
     &           para(1,1,kmdes),
     &           pare(1,1,kmdes),
     &           iterwmax,wrlx1,wrlx2,wrlx3,
     &           initwgt,initeng,iairf,
     &           ichoke5(1,kmdes),ichoke7(1,kmdes),
     &           Litprint,Lconv)
      call picwrt(lug,fnameg,pari,parg,parg(igxNP))

c---- calculate fuel-burn for off-design missions
      do km = 2, nmission
        if(parm(imwOpt,km) .eq. 0.0) then
c------- this off-design mission has no influence on objective function, 
c-       so skip its calculation for now to save CPU time
c-       (will calculate it after optimization converges)
         parm(imWfuel,km) = 0.

        else
         iairf = 2  !###
         call woper(pari,
     &              parg,
     &              parm(1,km),
     &              para(1,1,km),
     &              pare(1,1,km),
     &              para(1,1,kmdes),
     &              pare(1,1,kmdes),
     &              iterfmax,initeng,iairf,
     &              ichoke5(1,kmdes),ichoke7(1,kmdes),
     &              Litprint,Lconv)
        endif
      enddo

c---- set fleet fuel burn and Payload-Range
      Wftotal = 0.
      Hftotal = 0.
      PRtotal = 0.
      do km = 1, nmission
        wOpt = parm(imwOpt,km)
        if(wOpt .ne. 0.0) then
         hfuel = pare(iehfuel,ipcruise1,km)
         Wfuel = parm(imWfuel,km)
         Wburn = Wfuel/(1.0+parg(igfreserve))
         Wftotal = Wftotal + wOpt*Wfuel
         Hftotal = Hftotal + wOpt*Wburn*hfuel
         PRtotal = PRtotal + wOpt*parm(imWpay,km)*parm(imRange,km)
        endif
      enddo

c---- set fleet PFEI
      parg(igPFEI) = Hftotal/PRtotal

c---- set unconstrained objective function
      fun = Wftotal

c---- ad-hoc negative-sweep penalty to keep optimizer from jumping there
c-    ( model depends on cos(sweep) , so negative sweep is "invisible" )
      sweep = parg(igsweep)
      fcon = (1.5-sweep)/100.0
      fac = 1.0*parg(igWpay)
      fun = fun + fac*max( fcon , 0.0 )**2

c---- ad-hoc inner panel reverse taper penalty, which might otherwise
c-     look attractive to the optimizer for strut-wing cases,
c-     because it doesn't know about the download requirements
      lambdas = parg(iglambdas)
      fcon = (lambdas-1.0)/0.25
      fac = 1.0*parg(igWpay)
      fun = fun + fac*max( fcon , 0.0 )**2

c---- ad-hoc tip panel excessive taper penalty,
c-     to strongly discourage the optimizer from trying 
c-     a negative tip chord as it samples the design space.
      lambdat = parg(iglambdat)
      fcon = (0.15-lambdat)/0.25
      fac = 1.0*parg(igWpay)
      fun = fun + fac*max( fcon , 0.0 )**2

c-----------------------------------------------------------------------
c---- set constraint functions
      ncon = 0

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if(LlBFcon) then
c----- balanced-field constraint
       km = 1
       lBF    = parm(imlBF,km)
       lBFmax = parg(iglBFmax)

       ncon = ncon + 1
       con(ncon) = lBF/lBFmax - 1.0
       penfac(ncon) = 5.0 * parg(igWpay)
      endif

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if(LWfmaxcon) then
c----- fuel-volume constraint
       Wfuel  = parg(igWfuel)
       Wfmax  = parg(igWfmax)
       rWfmax = parg(igrWfmax)

       ncon = ncon + 1
       con(ncon) = Wfuel/(rWfmax*Wfmax) - 1.0
       penfac(ncon) = 5.0 * parg(igWpay)
      endif

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if(Lbmaxcon) then
c----- span constraint
       b  = parg(igb)
       bmax = parg(igbmax)

       ncon = ncon + 1
       con(ncon) = b/bmax - 1.0
       penfac(ncon) = 25.0 * parg(igWpay)
      endif

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if(Lgtoccon) then
c----- top-of-climb climb angle constraint
       ip = ipclimbn
       km = 1
       gtocmin = parg(iggtocmin)
       gtoc = para(iagamV,ip,km)
       ncon = ncon + 1
       con(ncon) = 1.0 - gtoc/gtocmin
       penfac(ncon) = 1.0 * parg(igWpay)
      endif

c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c---- set objective function with constraint penalties
      dfunc = 0.
      do icon = 1, ncon
        dfunc = dfunc + penfac(icon)*max(con(icon),0.0)**2
      enddo

      func = fun + dfunc

c---- store penalized objective function
      parg(igFOpt) = func

c---- INSTRUMENTATION: dump every objective evaluation -- the variable
c---- vector, the raw and penalised objective, and the constraints.
c---- Not part of TASOPT -- see fortran_ref/fobj_instrumented.f.
      open(83,file='fobj_calls.txt',status='unknown',position='append')
      write(83,'(a,i5,i4,i4)') 'CALL ', istep, nvopt, ncon
      write(83,'(e26.18)') (vopt(i), i=1,nvopt)
      write(83,'(e26.18)') fun, func
      if(ncon.gt.0) write(83,'(e26.18)') (con(i), i=1,ncon)
      write(83,'(e26.18)') parg(igPFEI), parg(igWMTO), parg(igWfuel)
      close(83)

c----------------------------------------------------------------------

c---- from now on, use current weight values as initial guesses
      initwgt = 1

c---- from now on, use current engine parameter values as initial guesses
      initeng = 1

      if(Loprint .and. istep.ge.0) then
c----- print quantities of interest, to monitor optimization progress

       if(mod(istep,10) .eq. 1) then
c------ print header every 10 optimization steps
c       write(*,1020)
c1020   format(1x,
c    & '   i   dFrel      Fobj      Wfuel     Wwing  '
c    & '    b       lBF     cd   ',
c    & '   CL     AR     sweep ',
c    & '  (t/c)o   (t/c)s   lams    lamt    rcls    rclt  ',
c    & ' Momax   Msmax    OPR     altkft   T4CR    T4TO  ',
c    & ' tbcapo  tbcaps' )

        write(*,1020)
 1020   format(1x,
     & '   i   dFrel      Fobj      Wfuel     WMTO   '
     & '    b       lBF     cd   ',
     & '   CL     AR     sweep ',
     & '  (t/c)o  (t/c)s  lams    lamt    rcls    rclt  ',
     & '   FPR     BPR    OPR      altft   T4CR    T4TO  ',
     & '  dfan  u6/u0_B1 u8/u0_B1 u6/u0_C1 u8/u0_C1' )
c       ?1234 123456789 12345678 12345678 1234567 1234567 1234567 
c       1234567 1234567 1234567 12345678 
c       12345678 12345678 1234567 1234567 1234567
       endif


c----- mission and point to be printed out
       km = 1

c       do 900 km = 1, nmission
c       if(parm(imwOpt,km) .eq. 0.0) go to 900

       ip = ipcruise1

c----- unpack stuff for printing
cc     Wfuel = parg(igWfuel)
       Wfuel = parm(imWfuel,km)

       WMTO  = parg(igWMTO)
       Wwing = parg(igWwing)
       AR    = parg(igAR)
       sweep = parg(igsweep)
       hboxo = parg(ighboxo)
       hboxs = parg(ighboxs)
       lams  = parg(iglambdas)
       lamt  = parg(iglambdat)
       b     = parg(igb)

       tbcapo = parg(igtbcapo)
       tbwebo = parg(igtbwebo)
       tbcaps = parg(igtbcaps)
       tbwebs = parg(igtbwebs)

       Momax = parg(igMomax)
       Msmax = parg(igMsmax)

       cdfw = para(iacdfw,ip,km)
       cdpw = para(iacdpw,ip,km)
       CL   = para(iaCL  ,ip,km)
       rcls = para(iarcls,ip,km)
       rclt = para(iarclt,ip,km)
       alt = para(iaalt,ip,km)
       pif = pare(iepif,ip,km)
       BPR = pare(ieBPR,ip,km)
       OPR = pare(iepilc,ip,km)*pare(iepihc,ip,km)
       dfan = parg(igdfan)

       Tt4CR = pare(ieTt4,ipcruise1,km)
       Tt4TO = pare(ieTt4,iptakeoff,km)
       thetaCB = parm(imthCB,km) * 180.0/pi

       funclbs = func*lb_N
       dfunclbs = dfunc*lb_N
       Wflbs = Wfuel*lb_N
       Wwlbs = Wwing*lb_N
       WMlbs = WMTO*lb_N
       ialtft = int(alt*ft_m + 0.5)
       bft = b*ft_m
       lBFft = parm(imlBF,km)*ft_m
       Molbs = Momax*lb_N*ft_m / 1.0e6
       Mslbs = Msmax*lb_N*ft_m / 1.0e6
       dfanin = dfan*in_m

       TSFC = pare(ieTSFC,ip,km)*3600.0
       DOL  = para(iaCD  ,ip,km)/CL

       u6ratc = pare(ieu6,ip,km)/pare(ieu0,ip,km)
       u8ratc = pare(ieu8,ip,km)/pare(ieu0,ip,km)

       ipb = ipclimb1
       u6ratb = pare(ieu6,ipb,km)/pare(ieu0,ipb,km)
       u8ratb = pare(ieu8,ipb,km)/pare(ieu0,ipb,km)

c      write(*,999) 0, dfrel, funclbs, (vopt(i), i=1, nvopt)
c 999  format(1x,i3, g12.4, 30g15.7)

       write(*,1100) ch, istep,
     &   dfrel, funclbs, Wflbs, WMlbs,
     &   bft, lBFft, cdfw+cdpw,
     &   CL, AR, sweep, hboxo,hboxs, lams,lamt, rcls,rclt,
     &   pif, BPR, OPR, ialtft, Tt4CR, Tt4TO,
     &   dfanin, u6ratb, u8ratb, u6ratc, u8ratc

 1100  format(1x,a, i4,
     &   g10.3, f10.2, f10.2, f10.1,
     &   f8.2, f8.1, f8.5,
     &   f8.5, f7.3, f8.3, 2f8.5, 2f8.4, 2f8.4, 
     &   f9.5, f8.4, f8.4, i8  , f8.1, f8.1,
     &   f8.3, f8.4, f8.4, 1X, f8.4, f8.4 )

      endif

c 900  continue

      return
      end ! fobj


      subroutine gobj(nvopt,vopt,dvopt,
     &                fun,funv, ncmax,ncon,con,conv)
c========================================================================
c     Evaluates objective function and gradient 
c     for optimization variables vopt(.)
c     Also determines constraint functions and gradients
c========================================================================
      implicit real (a-h,l-z)
      integer nvopt,
     &        ncmax,ncon

      real vopt(*), dvopt(*), funv(*),
     &     con(ncmax), conv(ncmax,*)
c
      include 'index.inc'
      include 'constants.inc'
      include 'airf.inc'
      include 'arrs.inc'
c
      real vopt1(iototal), con1(iototal),
     &     vopt2(iototal), con2(iototal)

      logical Lconv

c---- set function 
      dfrel = 0.
      call fobj(nvopt,vopt,ncon,con,fun,func,dfrel,' ',-1)

c---- set function derivatives for each optimization variable
      veps = 0.5
      do 110 io = 1, nvopt
        vdel = veps*dvopt(io)
        vopt1(io) = vopt(io) - 0.5*vdel
        vopt2(io) = vopt(io) + 0.5*vdel

        dfrel = 0.
        call fobj(nvopt,vopt1,ncon,con1,fun1,func,dfrel,' ',-1)
        call fobj(nvopt,vopt2,ncon,con2,fun2,func,dfrel,' ',-1)

        funv(io) = (fun2-fun1)/vdel
        do icon = 1, ncon
          conv(icon,io) = (con2(icon)-con1(icon))/vdel
        enddo
 110  continue

      return
      end ! gobj


      subroutine voptset(nvopt,vopt)
c---------------------------------------------------------------
c     Puts optimization variable values passed in via vopt(.)
c     into global variable arrays parg,para,pare, as directed
c     by the optimization-variable selection pointers iovar(.)
c---------------------------------------------------------------
      implicit real (a-h,l-z)
      integer nvopt
      real vopt(*)

      include 'index.inc'
      include 'constants.inc'
      include 'airf.inc'
      include 'arrs.inc'
c
      do iv = 1, nvopt
        io = iovar(iv)

        if(io .eq. ioCL   ) then
         do km = 1, nmission
           do ip = ipclimb1+1, ipdescentn-1
             para(iaCL,ip,km) = vopt(iv)
           enddo
         enddo
        endif

        if(io .eq. ioAR   ) parg(igAR   ) = vopt(iv)

        if(io .eq. iosweep) then
         parg(igsweep ) = vopt(iv)
         parg(igsweeph) = vopt(iv)
        endif

        if(io .eq. iohboxo) parg(ighboxo) = vopt(iv)
        if(io .eq. iohboxs) parg(ighboxs) = vopt(iv)
        if(io .eq. iolams ) parg(iglambdas) = vopt(iv)
        if(io .eq. iolamt ) then
         parg(iglambdat) = vopt(iv)
         if(parg(iglambdat) .lt. 0.1) then
          parg(iglambdat) = 0.1
          vopt(iv) = 0.1
         endif
        endif

        if(io .eq. iorcls ) then
         do km = 1, nmission
           do ip = ipclimb1+1, ipdescentn-1
             para(iarcls,ip,km) = vopt(iv)
           enddo
         enddo
        endif

        if(io .eq. iorclt ) then
         do km = 1, nmission
           do ip = ipclimb1+1, ipdescentn-1
             para(iarclt,ip,km) = vopt(iv)
           enddo
         enddo
        endif

        if(io .eq. ioFPR  ) then
         do km = 1, nmission
           do ip = ipcruise1, ipcruisen
             pare(iepif,ip,km) = vopt(iv)
           enddo
         enddo
        endif

        if(io .eq. ioBPR  ) then
         do km = 1, nmission
           do ip = ipcruise1, ipcruisen
             pare(ieBPR,ip,km) = vopt(iv)
           enddo
         enddo
        endif

        if(io .eq. ioalt  ) then
         do km = 1, nmission
           para(iaalt,ipcruise1,km) = vopt(iv)
         enddo
        endif

        if(io .eq. ioT4CR ) then
         do km = 1, nmission
           do ip = ipcruise1, ipcruisen
             pare(ieTt4,ip,km) = vopt(iv)
           enddo
         enddo
        endif

        if(io .eq. ioT4TO ) then
         do km = 1, nmission
           pare(ieTt4,ipstatic ,km) = vopt(iv)
           pare(ieTt4,iprotate ,km) = vopt(iv)
           pare(ieTt4,iptakeoff,km) = vopt(iv)
         enddo
        endif

        if(io .eq. ioOPR  ) then
         do km = 1, nmission
           do ip = ipcruise1, ipcruisen
             pare(iepihc,ip,km) = vopt(iv) / pare(iepilc,ip,km)
           enddo
         enddo
        endif

      enddo

      return
      end ! voptset
