      program drv_cdsum
c---- Drag buildup for a 737-class aircraft at cruise.
c---- Case 1: icdfun=0 (stored section cd's).
c---- Case 2: icdfun=1 (wing cd from the airfoil database).
c---- Case 3: as 2, with BLI credits switched on.
c---- Case 4: as 2, at zero lift (exercises cditrp's CL=0 branch).
      implicit real (a-h,l-z)
      include 'index.inc'
      include 'airf.inc'
      integer pari(iitotal), icase, iairf
      real parg(igtotal), para(iatotal), pare(ietotal)
      character*80 fname

      iairf = 1
      fname = '/Users/codykarcher/Desktop/Tasopt2.16/air/C.air'
      call airtable(fname, iAdim,jAdim,kAdim,lAdim,
     &  nAMa(iairf),nAcl(iairf),nAtau(iairf),nAfun(iairf),
     &  AMa(1,iairf),Acl(1,iairf),Atau(1,iairf),ARe(iairf),
     &          A(1,1,1,1,iairf),
     &        A_M(1,1,1,1,iairf),
     &       A_cl(1,1,1,1,iairf),
     &      A_tau(1,1,1,1,iairf),
     &     A_M_cl(1,1,1,1,iairf),
     &    A_M_tau(1,1,1,1,iairf),
     &   A_cl_tau(1,1,1,1,iairf),
     & A_M_cl_tau(1,1,1,1,iairf))

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

c---- wing
      parg(igS)       = 124.0
      parg(igb)       = 35.0
      parg(igbs)      = 10.5
      parg(igbo)      = 3.6
      parg(igco)      = 5.8
      parg(igAR)      = 35.0**2/124.0
      parg(igsweep)   = 26.0
      parg(iglambdat) = 0.18
      parg(iglambdas) = 0.70
      parg(iglambdac) = 1.0
      parg(ighboxo)   = 0.140
      parg(ighboxs)   = 0.126
      parg(igzwing)   = -1.2
      parg(igfLo)     = -0.3
      parg(igfLt)     = -0.05
c---- horizontal tail
      parg(igSh)      = 31.0
      parg(igbh)      = 13.0
      parg(igboh)     = 2.0
      parg(igcoh)     = 3.1
      parg(igARh)     = 13.0**2/31.0
      parg(igsweeph)  = 30.0
      parg(iglambdah) = 0.25
      parg(igzhtail)  = 1.2
      parg(igfCDhcen) = 0.1
c---- vertical tail
      parg(igSv)      = 27.0
      parg(igbv)      = 7.0
      parg(igbov)     = 0.0
      parg(igcov)     = 4.6
      parg(igARv)     = 7.0**2/27.0
      parg(igsweepv)  = 25.0
      parg(iglambdav) = 0.30
      parg(ignvtail)  = 1.0
c---- nacelle and strut
      parg(iglnace)   = 3.0
      parg(igfSnace)  = 8.5
      parg(igrVnace)  = 1.02
      parg(igSstrut)  = 0.0
      parg(igcosLs)   = 1.0
      parg(igrVstrut) = 1.0
c---- fuselage
      parg(igRfuse)   = 1.9
      parg(igxnose)   = 0.0
      parg(igxend)    = 37.0
c---- operating point
      para(iaCL)      = 0.57
      para(iaCLh)     = -0.05
      para(iaMach)    = 0.80
      para(iarcls)    = 1.238
      para(iarclt)    = 0.90
      para(iafduo)    = 0.018
      para(iafdus)    = 0.014
      para(iafdut)    = 0.0045
      para(iaReunit)  = 1.30e6
      para(iaRerefw)  = 2.0e7
      para(iaRereft)  = 1.0e7
      para(iaaRexp)   = -0.15
      para(iafexcdw)  = 1.02
      para(iafexcdt)  = 1.02
      para(iafexcdf)  = 1.03
      para(iacdfw)    = 0.0050
      para(iacdpw)    = 0.0035
      para(iacdft)    = 0.0060
      para(iacdpt)    = 0.0035
      para(iaPAfinf)  = 1.10
      para(iaDAfwake) = 0.30
      pare(ieM2)      = 0.60

      icdfun = 0
      if (icase .ge. 2) icdfun = 1
      if (icase .eq. 3) then
        parg(igfBLIf) = 0.4
        parg(igfBLIw) = 0.2
      endif
      if (icase .eq. 4) then
        para(iaCL)  = 0.0
        para(iaCLh) = 0.0
      endif

      call cdsum(pari,parg,para,pare, icdfun, iairf)

      write(*,900) icase,'CD    ',para(iaCD)
      write(*,900) icase,'CDi   ',para(iaCDi)
      write(*,900) icase,'spanef',para(iaspaneff)
      write(*,900) icase,'CDwing',para(iaCDwing)
      write(*,900) icase,'CDover',para(iaCDover)
      write(*,900) icase,'CDhtai',para(iaCDhtail)
      write(*,900) icase,'CDvtai',para(iaCDvtail)
      write(*,900) icase,'CDfuse',para(iaCDfuse)
      write(*,900) icase,'CDnace',para(iaCDnace)
      write(*,900) icase,'CDstru',para(iaCDstrut)
      write(*,900) icase,'Cfnace',para(iaCfnace)
      write(*,900) icase,'cdfw  ',para(iacdfw)
      write(*,900) icase,'cdpw  ',para(iacdpw)
      write(*,900) icase,'clpo  ',para(iaclpo)
      write(*,900) icase,'clps  ',para(iaclps)
      write(*,900) icase,'clpt  ',para(iaclpt)

 500  continue
 900  format(I2,',',A6,',',E24.16)
      stop
      end
