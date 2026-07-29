
      subroutine fusebl(pari,parg,para)
c========================================================================
c     Calculates surface velocities, boundary layer, wake 
c     for a quasi-axisymmetric body in compressible flow.
c
c     A compressible source line represents the potential flow.
c     An integral BL formulation with lateral divergence
c       represents the surface BL and wake.
c     An added-source distribution represents the viscous displacement
c       influence on the potential flow.
c
c     The body shape is defined by its area and perimeter distributions
c        A(x),  b0(x),
c     which are defined by the various geometric parameters in parg(.)
c========================================================================
      implicit real (a-h,l-z)

      include 'index.inc'
      integer pari(iitotal)
      real parg(igtotal), para(iatotal)

      include 'fbl.inc'
      include 'constants.inc'

      real Pend, Pinf, KTE, Kend, Kinf

      ifclose = pari(iifclose)

      xnose = parg(igxnose)
      xend  = parg(igxend )
      xblend1 = parg(igxblend1)
      xblend2 = parg(igxblend2)

      Mach = para(iaMach)
      altkm = para(iaalt)/1000.0
      call atmos(altkm, T0,p0,rho0,a0,mu0)
      Reunit = Mach*a0 * rho0/mu0

      wfb    = parg(igwfb)
      Rfuse  = parg(igRfuse)
      dRfuse = parg(igdRfuse)

c---- fuselage cross-section geometric parameters
      wfblim = max( min( wfb , Rfuse ) , 0.0 )
      thetafb = asin(wfblim/Rfuse)
      hfb = sqrt(Rfuse**2 - wfb**2)
      sin2t = 2.0*hfb*wfb/Rfuse**2
      Sfuse = (pi + 2.0*thetafb + sin2t)*Rfuse**2 + 2.0*Rfuse*dRfuse

      anose = parg(iganose)
      btail = parg(igbtail)

c---- calculate potential-flow surface velocity uinv(.) using PG source line
      nc = 30
      call axisol(xnose,xend,xblend1,xblend2,Sfuse, 
     &            anose,btail,ifclose,
     &            Mach, nc,
     &            nbldim, nbl, iblte, xbl,zbl,sbl,dybl,uinv)

c---- fuselage perimeter
      if(ifclose.eq.0) then
       do ibl = 1, nbl
         bbl(ibl) = 2.0*pi*zbl(ibl)
       enddo
      else
       do ibl = 1, nbl
         bbl(ibl) = 2.0*pi*zbl(ibl) + 4.0*dybl(ibl)
       enddo
      endif

c---- dr/dn  (cosine of body contour angle from axis)
      do ibl = 2, nbl-1
        dxm = xbl(ibl) - xbl(ibl-1)
        dzm = zbl(ibl) - zbl(ibl-1)
        dsm = sbl(ibl) - sbl(ibl-1)

        dxp = xbl(ibl+1) - xbl(ibl)
        dzp = zbl(ibl+1) - zbl(ibl)
        dsp = sbl(ibl+1) - sbl(ibl)

        dxo = (dxm/dsm)*dsp + (dxp/dsp)*dsm
        dzo = (dzm/dsm)*dsp + (dzp/dsp)*dsm

        rnbl(ibl) = dxo / sqrt(dxo**2 + dzo**2)
      enddo
c---- extrapolate to endpoints
      rnbl(1) = rnbl(2)
     &        - (0.5*(sbl(3)+sbl(2)) - sbl(1))
     &         *(rnbl(3)-rnbl(2))/(sbl(3)-sbl(2))
      rnbl(nbl) = rnbl(nbl-1)
     &        + (sbl(nbl) - 0.5*(sbl(nbl-1)+sbl(nbl-2)))
     &         *(rnbl(nbl-1)-rnbl(nbl-2))/(sbl(nbl-1)-sbl(nbl-2))
      rnbl(1) = max( rnbl(1) , 0.0 )
      rnbl(nbl) = max( rnbl(nbl) , 0.0 )

c---- perform viscous/inviscid BL calculation driven by uinv(.)
      fex = para(iafexcdf)
      call blax(nbldim, nbl,iblte, 
     &          sbl, bbl, rnbl, uinv, Reunit, Mach, fex,
     &          uebl, dsbl, thbl, tsbl, dcbl,
     &          cfbl, cdbl, ctbl, hkbl, phbl )

      gam = gamSL
      gmi = gam - 1.0

c---- KE defect at TE and surface dissipation
      i = iblte
      trbl = 1.0 + 0.5*gmi*Mach**2*(1.0 - uebl(i)**2) 
      rhbl = trbl**(1.0/gmi)
      KTE = 0.5*rhbl*uebl(i)**3 * tsbl(i) * (bbl(i) + 2.0*pi*dsbl(i))
      Difsurf = phbl(i)

c---- momentum and KE defects and accumulated dissipation at end of wake
      i = nbl
      trbl = 1.0 + 0.5*gmi*Mach**2*(1.0 - uebl(i)**2) 
      rhbl = trbl**(1.0/gmi)
      Pend =     rhbl*uebl(i)**2 * thbl(i) * (bbl(i) + 2.0*pi*dsbl(i))
      Kend = 0.5*rhbl*uebl(i)**3 * tsbl(i) * (bbl(i) + 2.0*pi*dsbl(i))
      Difend = phbl(i)

c---- far-downstream momentum defect via Squire-Young (note that Vinf = 1 here)
      Hend = dsbl(i)/thbl(i)
      Hinf = 1.0 + gmi*Mach**2
      Havg = 0.5*(Hend+Hinf)
      Pinf = Pend * uebl(i)**Havg

c---- far-downstream KE defect  0.5 rho V^3 Theta*  (note that Vinf = 1 here)
      Kinf = Pinf

c---- additional dissipation downstream of last wake point
      tsinf = 2.0*Kinf
      dcinf = 0.5*gmi*Mach**2 * tsinf
      dDif = Kinf - Kend 
     &     + 0.5*(dcinf + dcbl(i)) * (1.0 - uebl(i))
      Difinf = Difend + dDif

c---- wake dissipation
      Difwake = Difinf - Difsurf

c---- store dissipation, KE, and drag areas
      Vinf = 1.0
      qinf = 0.5
      para(iaDAfsurf) = Difsurf/(qinf*Vinf)
      para(iaDAfwake) = Difwake/(qinf*Vinf)
      para(iaKAfTE)   = KTE/(qinf*Vinf)
      para(iaPAfinf)  = Pinf/qinf

c---- INSTRUMENTATION: dump the complete input state this call read, and the
c---- four outputs it produced, for every fusebl call in a real sizing run.
c---- Not part of TASOPT -- see fortran_ref/fusebl_instrumented.f.
      open(unit=87, file='fusebl_calls.csv', status='unknown',
     &     position='append')
      write(87,8701) ifclose, xnose, xend, xblend1, xblend2,
     &   anose, btail, Rfuse, dRfuse, wfb, Mach, para(iaalt), fex,
     &   para(iaDAfsurf), para(iaDAfwake), para(iaKAfTE), para(iaPAfinf)
 8701 format(I2,',',15(E26.18,','),E26.18)
      close(87)

      return
      end ! fusebl

