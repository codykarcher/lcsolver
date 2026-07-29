      program drv_fusebl
c---- Fuselage BL driver, on the shipped 737 fuselage (737.tas geometry) at a
c---- range of Mach numbers, altitudes and cross-sections.
c----
c---- Cases:
c----   1  737 as shipped:  M = 0.80, 35 kft, round section, fexcdf = 1.03
c----   2  same, smooth wall (fexcdf = 1.0)
c----   3  M = 0.60 at 20 kft
c----   4  M = 0.20 at sea level
c----   5  double bubble: wfb = 0.4 m, dRfuse doubled
c----   6  double bubble with an edge-type tail (ifclose = 1)
c----   7  fatter, blunter fuselage
c----
c---- gfortran -fdefault-real-8 -O0 -fdollar-ok -I. -o drv_fusebl \
c----          drv_fusebl.f fusebl.f blax.f blsys.f axisol.f gaussn.f atmos.f
      implicit real (a-h,m,o-z)

      include 'index.inc'
      integer pari(iitotal)
      real parg(igtotal), para(iatotal)

      include 'fbl.inc'
      include 'constants.inc'

      integer icase, i

c---- constants.inc is a COMMON block filled at runtime by tasopt.f
      pi    = 3.1415926535897932384626
      gee   = 9.81
      cpSL  = 1004.0
      gamSL = 1.4

      write(*,'(A)') 'case,DAfsurf,DAfwake,KAfTE,PAfinf'

      do 500 icase = 1, 7

        do i = 1, iitotal
          pari(i) = 0
        enddo
        do i = 1, igtotal
          parg(i) = 0.0
        enddo
        do i = 1, iatotal
          para(i) = 0.0
        enddo

        pari(iifclose) = 0

        parg(igxnose)   =   0.0 * 0.3048
        parg(igxend)    = 124.0 * 0.3048
        parg(igxblend1) =  20.0 * 0.3048
        parg(igxblend2) =  97.0 * 0.3048
        parg(iganose)   = 1.65
        parg(igbtail)   = 2.0
        parg(igRfuse)   = 77.0 * 0.0254
        parg(igdRfuse)  = 15.0 * 0.0254
        parg(igwfb)     = 0.0

        para(iaMach)   = 0.80
        para(iaalt)    = 35000.0 * 0.3048
        para(iafexcdf) = 1.03

        if(icase.eq.2) para(iafexcdf) = 1.0
        if(icase.eq.3) then
         para(iaMach) = 0.60
         para(iaalt)  = 20000.0 * 0.3048
        endif
        if(icase.eq.4) then
         para(iaMach) = 0.20
         para(iaalt)  = 0.0
        endif
        if(icase.eq.5) then
         parg(igwfb)    = 0.40
         parg(igdRfuse) = 30.0 * 0.0254
        endif
        if(icase.eq.6) then
         pari(iifclose) = 1
         parg(igwfb)    = 0.40
         parg(igdRfuse) = 30.0 * 0.0254
        endif
        if(icase.eq.7) then
         parg(igRfuse) = 90.0 * 0.0254
         parg(iganose) = 2.20
         parg(igbtail) = 1.40
        endif

        call fusebl(pari,parg,para)

        write(*,901) icase, para(iaDAfsurf), para(iaDAfwake),
     &               para(iaKAfTE), para(iaPAfinf)

 500  continue
 901  format(I2,',',3(E26.18,','),E26.18)
      stop
      end
