      program drv_surfcd
c---- Exercises surfcd (the closed-form spanwise integral) across the two
c---- taper-ratio branches. Cases 4 and 5 sit inside the 0.02 windows where
c---- the routine switches to its asymptotic expansions, which is where a
c---- port is most likely to disagree.
      implicit real (a-z)
      integer icase
      real AMa(1), Acl(1), Atau(1)
      real A(1), A_M(1), A_cl(1), A_tau(1)
      real A_M_cl(1), A_M_tau(1), A_cl_tau(1), A_M_cl_tau(1)
      write(*,'(A)') 'case,name,value'
      do icase = 1, 6
        S = 105.0d0
        b = 35.0d0
        bs = 12.0d0
        bo = 3.6d0
        lambdat = 0.25d0
        lambdas = 0.65d0
        sweep = 26.0d0
        co = 5.6d0
        cdf = 0.0050d0
        cdp = 0.0025d0
        Reco = 2.0d7
        Reref = 1.0d7
        aRexp = -0.15d0
        kSuns = 0.5d0
        fCDcen = 1.0d0
        if (icase .eq. 2) then
c------- unswept
          sweep = 0.0d0
        endif
        if (icase .eq. 3) then
c------- highly swept, sharper taper
          sweep = 35.0d0
          lambdat = 0.15d0
          lambdas = 0.50d0
          Reco = 3.5d7
        endif
        if (icase .eq. 4) then
c------- lambdas within 0.02 of 1: asymptotic lsfac branch
          lambdas = 0.995d0
          lambdat = 0.30d0
        endif
        if (icase .eq. 5) then
c------- lambdat within 0.02 of lambdas: asymptotic lfac branch
          lambdas = 0.60d0
          lambdat = 0.59d0
        endif
        if (icase .eq. 6) then
c------- both asymptotic branches at once, and a positive Re exponent
          lambdas = 0.99d0
          lambdat = 0.985d0
          aRexp = 0.20d0
          fCDcen = 0.6d0
        endif
        call surfcd(S,
     &    b,bs,bo,lambdat,lambdas,sweep,co,
     &    cdf,cdp,Reco,Reref,aRexp, kSuns,
     &    fCDcen,
     &    CDsurf,CDover)
        write(*,'(I2,A,A,A,E24.16)') icase,',','CDsurf',',',CDsurf
        write(*,'(I2,A,A,A,E24.16)') icase,',','CDover',',',CDover
      enddo

c---- surfcd2: the spanwise quadrature, exercised against a deterministic
c---- stand-in for airfun (see airfun_stub.f) so the integration and the
c---- sweep/unsweep factors are verified without the spline tables.
      do icase = 7, 9
        S = 105.0d0
        b = 35.0d0
        bs = 12.0d0
        bo = 3.6d0
        lambdat = 0.25d0
        lambdas = 0.65d0
        gammat = 0.22d0
        gammas = 0.70d0
        toco = 0.13d0
        tocs = 0.12d0
        toct = 0.10d0
        Mach = 0.80d0
        sweep = 26.0d0
        co = 5.6d0
        CL = 0.55d0
        CLhtail = -0.05d0
        fLo = -0.3d0
        fLt = -0.05d0
        Reco = 2.0d7
        aRexp = -0.15d0
        kSuns = 0.5d0
        fexcd = 1.0d0
        ARe = 1.0d7
        fduo = 0.0d0
        fdus = 0.0d0
        fdut = 0.0d0
        if (icase .eq. 8) then
          fduo = 0.018d0
          fdus = 0.014d0
          fdut = 0.009d0
          Mach = 0.72d0
        endif
        if (icase .eq. 9) then
          sweep = 35.0d0
          lambdat = 0.15d0
          gammat = 0.10d0
          CL = 0.70d0
          CLhtail = 0.0d0
          fexcd = 1.03d0
        endif
        call surfcd2(
     &    S, b,bs,bo,
     &    lambdat,lambdas,gammat,gammas,
     &    toco,tocs,toct,
     &    Mach,sweep,co,
     &    CL,CLhtail, fLo,fLt,
     &    Reco,aRexp, kSuns,fexcd,
     &    1,1,1,1, 1,1,1,1,
     &    AMa,Acl,Atau,ARe,
     &    A, A_M, A_cl, A_tau,
     &    A_M_cl, A_M_tau, A_cl_tau, A_M_cl_tau,
     &    fduo,fdus,fdut,
     &    clpo,clps,clpt,
     &    CDfwing,CDpwing,CDwing,CDover)
        write(*,'(I2,A,A,A,E24.16)') icase,',','clpo',',',clpo
        write(*,'(I2,A,A,A,E24.16)') icase,',','clps',',',clps
        write(*,'(I2,A,A,A,E24.16)') icase,',','clpt',',',clpt
        write(*,'(I2,A,A,A,E24.16)') icase,',','CDfwing',',',CDfwing
        write(*,'(I2,A,A,A,E24.16)') icase,',','CDpwing',',',CDpwing
        write(*,'(I2,A,A,A,E24.16)') icase,',','CDwing',',',CDwing
      enddo
      stop
      end
