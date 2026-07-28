      subroutine airfun(cl,toc,Mperp,
     &            iAdim,jAdim,kAdim,lAdim,
     &            nAMa,nAcl,nAtau,nAfun,
     &            AMa,Acl,Atau,ARe,
     &            A, A_M, A_cl, A_tau,
     &            A_M_cl, A_M_tau, A_cl_tau, A_M_cl_tau,
     &            cdf, cdp, cdwbar, cm)
c---- Deterministic stand-in for the airfoil database, so that surfcd2's
c---- quadrature can be verified independently of the spline tables. The
c---- Python test uses the identical formula.
      implicit real (a-z)
      integer iAdim,jAdim,kAdim,lAdim
      integer nAMa,nAcl,nAtau,nAfun
      real AMa(*), Acl(*), Atau(*), ARe
      real A(*), A_M(*), A_cl(*), A_tau(*)
      real A_M_cl(*), A_M_tau(*), A_cl_tau(*), A_M_cl_tau(*)
      cdf = 0.0040d0 + 0.0100d0*toc + 0.00050d0*cl**2
      cdp = 0.0020d0 + 0.0300d0*toc**2 + 0.0030d0*Mperp**4
     &    + 0.0010d0*cl**2
      cdwbar = 0.0d0
      cm = -0.10d0
      return
      end
