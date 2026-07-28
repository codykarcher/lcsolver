      program drv_balance
c---- Trim, CG limits and tail sizing for a 737-class aircraft.
      implicit real (a-h,l-z)
      include 'index.inc'
      integer pari(iitotal), icase, itrim, i
      real parg(igtotal), para(iatotal)
      real paraF(iatotal), paraB(iatotal), paraC(iatotal)
      include 'constants.inc'

c---- constants.inc is a COMMON block that tasopt.f fills at startup; a
c---- standalone driver must fill it too or balance sees pi = 0.
      pi  = 3.1415926535897932384626
      gee = 9.81
      cpSL  = 1004.0
      gamSL = 1.4
      write(*,'(A)') 'case,name,value'
      do 500 icase = 1, 6

      do i = 1, iitotal
        pari(i) = 0
      enddo
      do i = 1, igtotal
        parg(i) = 0.0
      enddo
      do i = 1, iatotal
        para(i)  = 0.0
        paraF(i) = 0.0
        paraB(i) = 0.0
        paraC(i) = 0.0
      enddo

      pari(iiengloc) = 1
      parg(igWMTO)   = 7.5e5
      parg(igWpay)   = 1.7e5
      parg(igWfuel)  = 1.6e5
      parg(igWfuse)  = 1.9e5
      parg(igWwing)  = 9.5e4
      parg(igWstrut) = 0.0
      parg(igWhtail) = 1.4e4
      parg(igWvtail) = 1.1e4
      parg(igWeng)   = 5.3e4
      parg(igfhpesys) = 0.010
      parg(igflgnose) = 0.011
      parg(igflgmain) = 0.044
      parg(igxshell1) = 5.2
      parg(igxshell2) = 31.0
      parg(igxwbox)   = 17.0
      parg(igxwing)   = 17.6
      parg(igxhbox)   = 34.0
      parg(igxvbox)   = 33.0
      parg(igxhtail)  = 34.6
      parg(igxvtail)  = 33.5
      parg(igxeng)    = 15.0
      parg(igxhpesys) = 19.0
      parg(igxlgnose) = 5.0
      parg(igdxlgmain) = 0.8
      parg(igxWfuse)  = 3.4e6
      parg(igdxWfuel)  = 0.0
      parg(igdxWwing)  = 2.0e4
      parg(igdxWstrut) = 0.0
      parg(igdxWhtail) = 1.5e3
      parg(igdxWvtail) = 1.2e3
      parg(igS)   = 124.0
      parg(igSh)  = 31.0
      parg(igco)  = 5.8
      parg(igcoh) = 3.1
      parg(igcma) = 4.2
      parg(igsweep) = 26.0
      parg(igCMVf1) = 60.0
      parg(igCLMf0) = 0.185
      parg(igdCLhdCL) = 0.55
      parg(igdCLndCL) = 0.015
      parg(igneng) = 2.0
      parg(igdfan) = 1.7
      parg(igVh)   = 1.45
      parg(igSMmin) = 0.05
      parg(igCLhspec)  = -0.02
      parg(igCLhCGfwd) = -0.65

      para(iaCMw0) = -0.09
      para(iaCMw1) = 0.015
      para(iaCMh0) = -0.02
      para(iaCMh1) = -0.30
      para(iaCL)   = 0.57
      para(iaCLh)  = -0.05

      do i = 1, iatotal
        paraF(i) = para(i)
        paraB(i) = para(i)
        paraC(i) = para(i)
      enddo
      paraF(iaclpmax) = 1.25
      paraC(iafracW)  = 0.90

      if (icase .le. 4) then
        itrim = icase - 1
        call balance(pari,parg,para, 0.65, 1.0, 0.5, itrim)
        write(*,900) icase,'xCG  ',para(iaxCG)
        write(*,900) icase,'xCP  ',para(iaxCP)
        write(*,900) icase,'xNP  ',para(iaxNP)
        write(*,900) icase,'CLh  ',para(iaCLh)
        write(*,900) icase,'Sh   ',parg(igSh)
        write(*,900) icase,'xwbox',parg(igxwbox)
        write(*,900) icase,'xwing',parg(igxwing)
        call cglpay(parg, rfF,rpF,xcF, rfB,rpB,xcB)
        write(*,900) icase,'rpayF',rpF
        write(*,900) icase,'xcgF ',xcF
        write(*,900) icase,'rpayB',rpB
        write(*,900) icase,'xcgB ',xcB
      else
        if (icase .eq. 5) then
          pari(iiHTsize) = 1
          pari(iixwmove) = 1
        else
          pari(iiHTsize) = 2
          pari(iixwmove) = 2
        endif
        call htsize(pari,parg,paraF,paraB,paraC)
        write(*,900) icase,'Sh   ',parg(igSh)
        write(*,900) icase,'xwbox',parg(igxwbox)
        write(*,900) icase,'xwing',parg(igxwing)
        write(*,900) icase,'xCGfw',parg(igxCGfwd)
        write(*,900) icase,'xCGaf',parg(igxCGaft)
        write(*,900) icase,'CLhfw',parg(igCLhCGfwd)
      endif

 500  continue
 900  format(I2,',',A5,',',E24.16)
      stop
      end

      subroutine compare(ss,aa,dd)
      character*2 ss
      real aa, dd
      ss = '  '
      return
      end
