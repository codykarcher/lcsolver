      program drv_takeoff
c---- Takeoff and balanced field length for a 737-class aircraft.
c---- Case 1: normal.  Case 2: heavier (longer field).
c---- Case 3: thrust cut so the engine-out case is impossible.
      implicit real (a-h,k-z)
      include 'index.inc'
      include 'airf.inc'
      include 'constants.inc'
      integer pari(iitotal), icase, i, ip, initeng, iairf
      integer ichoke5(iptotal), ichoke7(iptotal)
      real parg(igtotal), parm(imtotal)
      real para(iatotal,iptotal), pare(ietotal,iptotal)
      character*80 fname

      pi  = 3.1415926535897932384626
      gee = 9.81
      cpSL = 1004.0
      gamSL = 1.4

      iairf = 1
      fname = '/Users/codykarcher/Desktop/Tasopt2.16/air/C.air'
      call airtable(fname, iAdim,jAdim,kAdim,lAdim,
     &  nAMa(iairf),nAcl(iairf),nAtau(iairf),nAfun(iairf),
     &  AMa(1,iairf),Acl(1,iairf),Atau(1,iairf),ARe(iairf),
     &          A(1,1,1,1,iairf),        A_M(1,1,1,1,iairf),
     &       A_cl(1,1,1,1,iairf),      A_tau(1,1,1,1,iairf),
     &     A_M_cl(1,1,1,1,iairf),    A_M_tau(1,1,1,1,iairf),
     &   A_cl_tau(1,1,1,1,iairf), A_M_cl_tau(1,1,1,1,iairf))

      write(*,'(A)') 'case,name,value'
      do 500 icase = 1, 3

      do i = 1, iitotal
        pari(i) = 0
      enddo
      do i = 1, igtotal
        parg(i) = 0.0
      enddo
      do i = 1, imtotal
        parm(i) = 0.0
      enddo
      do ip = 1, iptotal
        do i = 1, iatotal
          para(i,ip) = 0.0
        enddo
        do i = 1, ietotal
          pare(i,ip) = 0.0
        enddo
      enddo

c---- geometry, as in the cdsum driver
      parg(igS)       = 124.0
      parg(igb)       = 35.0
      parg(igbs)      = 10.5
      parg(igbo)      = 3.6
      parg(igco)      = 5.8
      parg(igAR)      = 35.0**2/124.0
      parg(igsweep)   = 26.0
      parg(iglambdat) = 0.18
      parg(iglambdas) = 0.70
      parg(ighboxo)   = 0.140
      parg(ighboxs)   = 0.126
      parg(igzwing)   = -1.2
      parg(igfLo)     = -0.3
      parg(igfLt)     = -0.05
      parg(igSh)      = 31.0
      parg(igbh)      = 13.0
      parg(igboh)     = 2.0
      parg(igcoh)     = 3.1
      parg(igsweeph)  = 30.0
      parg(iglambdah) = 0.25
      parg(igzhtail)  = 1.2
      parg(igfCDhcen) = 0.1
      parg(igSv)      = 27.0
      parg(igbv)      = 7.0
      parg(igbov)     = 0.0
      parg(igcov)     = 4.6
      parg(igsweepv)  = 25.0
      parg(iglambdav) = 0.30
      parg(ignvtail)  = 1.0
      parg(iglnace)   = 3.0
      parg(igfSnace)  = 8.5
      parg(igrVnace)  = 1.02
      parg(igcosLs)   = 1.0
      parg(igrVstrut) = 1.0
c---- takeoff-specific
      parg(igWMTO)    = 7.5e5
      parg(igWfuel)   = 1.6e5
      parg(igWpay)    = 1.7e5
      parg(igdfan)    = 1.7
      parg(igHTRf)    = 0.30
      parg(igneng)    = 2.0
      parg(igmuroll)  = 0.025
      parg(igmubrake) = 0.35
      parg(ighobst)   = 10.7
      parg(igCDgear)  = 0.015
      parg(igcdefan)  = 0.10
      parg(igCDspoil) = 0.10
      parm(imWpay)    = 1.7e5
      parm(imWTO)     = 7.5e5

      do ip = 1, iptotal
        para(iaCL,ip)     = 0.57
        para(iaCLh,ip)    = -0.05
        para(iaMach,ip)   = 0.20
        para(iarcls,ip)   = 1.238
        para(iarclt,ip)   = 0.90
        para(iafduo,ip)   = 0.018
        para(iafdus,ip)   = 0.014
        para(iafdut,ip)   = 0.0045
        para(iaReunit,ip) = 2.5e6
        para(iaRerefw,ip) = 2.0e7
        para(iaRereft,ip) = 1.0e7
        para(iaaRexp,ip)  = -0.15
        para(iafexcdw,ip) = 1.02
        para(iafexcdt,ip) = 1.02
        para(iafexcdf,ip) = 1.03
        para(iacdfw,ip)   = 0.0050
        para(iacdpw,ip)   = 0.0035
        para(iacdft,ip)   = 0.0060
        para(iacdpt,ip)   = 0.0035
        para(iaPAfinf,ip) = 1.10
        pare(ieM2,ip)     = 0.60
        pare(ierho0,ip)   = 1.225
      enddo
      para(iaCD,ipclimb1) = 0.045

      pare(ieu0,iprotate)  = 68.0
      pare(ieu0,iptakeoff) = 78.0
      pare(ieFe,ipstatic)  = 1.10e5
      pare(ieFe,iprotate)  = 0.92e5
      pare(iemcore,ipstatic) = 45.0
      pare(iemcore,iprotate) = 43.0
      pare(ieff,ipstatic)  = 0.030
      pare(ieff,iprotate)  = 0.028

      if (icase .eq. 2) then
        parm(imWTO) = 9.5e5
      endif
      if (icase .eq. 3) then
        pare(ieFe,ipstatic) = 3.3e4
        pare(ieFe,iprotate) = 2.8e4
      endif

      initeng = 0
      call takeoff(pari,parg,parm,para,pare, initeng, ichoke5,ichoke7)

      write(*,900) icase,'V1   ',parm(imV1)
      write(*,900) icase,'V2   ',parm(imV2)
      write(*,900) icase,'lTO  ',parm(imlTO)
      write(*,900) icase,'l1   ',parm(iml1)
      write(*,900) icase,'lBF  ',parm(imlBF)
      write(*,900) icase,'tTO  ',parm(imtTO)
      write(*,900) icase,'FTO  ',parm(imFTO)
      write(*,900) icase,'gamTO',parm(imgamVTO)
      write(*,900) icase,'gamBF',parm(imgamVBF)
      write(*,900) icase,'fracW',para(iafracW,ipstatic)
      write(*,900) icase,'time ',para(iatime,ipstatic)
      write(*,900) icase,'Range',para(iaRange,ipstatic)

 500  continue
 900  format(I2,',',A5,',',E24.16)
      stop
      end
