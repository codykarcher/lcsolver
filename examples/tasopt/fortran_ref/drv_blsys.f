      program drv_blsys
c---- blvar (station values + derivatives) and blsys (the 2-point 3x3
c---- system), over ten regimes x five states each.
c----
c---- Cases:
c----   1  turbulent, attached, direct mode
c----   2  turbulent, attached, inverse mode (Hk held at hksep)
c----   3  wake  (also pins the cd_ue/cf_ue typo: cf_ue is preset below
c----             and blvar's wake branch must leave it alone)
c----   4  similarity station (simi)
c----   5  laminar
c----   6  turbulent, separated branch of HST (large H)
c----   7  turbulent at low Rtheta -- HST's Rt<400 and Rt<200 floors
c----   8  laminar at large H -- HSL, CFL and DIL second branches
c----   9  small H -- blvar's hk = max(hk,1.005) clamp
c----  10  large H at very low Rt -- CFT's arg and grt clamps
c----
c---- gfortran -fdefault-real-8 -O0 -o drv_blsys drv_blsys.f blsys.f
      implicit real (a-h,m,o-z)
      integer icase, j
      logical simi, lami, wake, direct
      real aa(3,3), bb(3,3), rr(3)

      write(*,'(A)') 'case,j,name,value'

      do 500 icase = 1, 10
      do 400 j = 1, 5

        simi   = .false.
        lami   = .false.
        wake   = .false.
        direct = .true.

        Reyn  = 2.0e7
        rMach = 0.80
        fexcr = 1.03
        hksep = 2.9
        uinv  = 1.04

        x   = 10.0
        b   = 6.0
        rn  = 0.90
        xm  = 9.0
        bm  = 5.8
        rnm = 0.88

        th = 0.02 + 0.02*float(j)
        ds = th*(1.6 + 0.4*float(j))
        ue = 0.80 + 0.10*float(j)

        if(icase.eq.2) direct = .false.
        if(icase.eq.3) wake = .true.
        if(icase.eq.4) simi = .true.
        if(icase.eq.5) then
         lami = .true.
         Reyn = 1.0e6
        endif
        if(icase.eq.6) ds = th*(3.0 + 0.6*float(j))
        if(icase.eq.7) then
         Reyn = 1.0e4
         th   = 0.010 + 0.002*float(j)
         ds   = th*(1.8 + 0.3*float(j))
        endif
        if(icase.eq.8) then
         lami = .true.
         Reyn = 1.0e6
         ds   = th*(5.0 + 0.8*float(j))
        endif
        if(icase.eq.9) ds = th*(0.9 + 0.02*float(j))
        if(icase.eq.10) then
         Reyn = 5.0e2
         ds   = th*(18.0 + 1.0*float(j))
        endif

        thm = th*0.93
        dsm = ds*0.90
        uem = ue*0.98
        if(simi) then
         thm = 0.
         dsm = 0.
         uem = 0.
        endif

c------ upstream station.  For simi it is unused by blsys, so zero it out
c------ rather than leave it to whatever happens to be lying about.
        hm     = 0.
        hm_thm = 0.
        hm_dsm = 0.
        hkm     = 0.
        hkm_thm = 0.
        hkm_dsm = 0.
        hkm_uem = 0.
        hcm     = 0.
        hcm_thm = 0.
        hcm_dsm = 0.
        hcm_uem = 0.
        hsm     = 0.
        hsm_thm = 0.
        hsm_dsm = 0.
        hsm_uem = 0.
        cfm     = 0.
        cfm_thm = 0.
        cfm_dsm = 0.
        cfm_uem = 0.
        dim     = 0.
        dim_thm = 0.
        dim_dsm = 0.
        dim_uem = 0.

        if(.not.simi) then
         cfm_uem = 0.
         call blvar(.false.,lami,.false., Reyn,rMach, fexcr,
     &              xm, thm ,dsm ,uem ,
     &              hm , hm_thm, hm_dsm,
     &              hkm, hkm_thm, hkm_dsm, hkm_uem,
     &              hcm, hcm_thm, hcm_dsm, hcm_uem,
     &              hsm, hsm_thm, hsm_dsm, hsm_uem,
     &              cfm, cfm_thm, cfm_dsm, cfm_uem,
     &              dim, dim_thm, dim_dsm, dim_uem )
        endif

c------ preset cf_ue to a recognisable value.  blvar's wake branch writes
c------ cd_ue instead of cf_ue, so on case 3 this value must come back out.
        cf_ue = 0.1234567890123456

        call blvar(simi,lami,wake, Reyn,rMach, fexcr,
     &             x, th ,ds ,ue ,
     &             h , h_th, h_ds,
     &             hk, hk_th, hk_ds, hk_ue,
     &             hc, hc_th, hc_ds, hc_ue,
     &             hs, hs_th, hs_ds, hs_ue,
     &             cf, cf_th, cf_ds, cf_ue,
     &             di, di_th, di_ds, di_ue )

        call blsys(simi,lami,wake,direct, rMach, uinv,hksep,
     &             x,b,rn,th,ds,ue,
     &             h , h_th, h_ds,
     &             hk, hk_th, hk_ds, hk_ue,
     &             hc, hc_th, hc_ds, hc_ue,
     &             hs, hs_th, hs_ds, hs_ue,
     &             cf, cf_th, cf_ds, cf_ue,
     &             di, di_th, di_ds, di_ue,
     &             xm,bm,rnm,thm,dsm,uem,
     &             hm , hm_thm, hm_dsm,
     &             hkm, hkm_thm, hkm_dsm, hkm_uem,
     &             hcm, hcm_thm, hcm_dsm, hcm_uem,
     &             hsm, hsm_thm, hsm_dsm, hsm_uem,
     &             cfm, cfm_thm, cfm_dsm, cfm_uem,
     &             dim, dim_thm, dim_dsm, dim_uem,
     &             aa,bb,rr)

        write(*,900) icase,j,'h     ', h
        write(*,900) icase,j,'h_th  ', h_th
        write(*,900) icase,j,'h_ds  ', h_ds
        write(*,900) icase,j,'hk    ', hk
        write(*,900) icase,j,'hk_th ', hk_th
        write(*,900) icase,j,'hk_ds ', hk_ds
        write(*,900) icase,j,'hk_ue ', hk_ue
        write(*,900) icase,j,'hc    ', hc
        write(*,900) icase,j,'hc_th ', hc_th
        write(*,900) icase,j,'hc_ds ', hc_ds
        write(*,900) icase,j,'hc_ue ', hc_ue
        write(*,900) icase,j,'hs    ', hs
        write(*,900) icase,j,'hs_th ', hs_th
        write(*,900) icase,j,'hs_ds ', hs_ds
        write(*,900) icase,j,'hs_ue ', hs_ue
        write(*,900) icase,j,'cf    ', cf
        write(*,900) icase,j,'cf_th ', cf_th
        write(*,900) icase,j,'cf_ds ', cf_ds
        write(*,900) icase,j,'cf_ue ', cf_ue
        write(*,900) icase,j,'di    ', di
        write(*,900) icase,j,'di_th ', di_th
        write(*,900) icase,j,'di_ds ', di_ds
        write(*,900) icase,j,'di_ue ', di_ue

        write(*,900) icase,j,'rr1   ', rr(1)
        write(*,900) icase,j,'rr2   ', rr(2)
        write(*,900) icase,j,'rr3   ', rr(3)
        write(*,900) icase,j,'aa11  ', aa(1,1)
        write(*,900) icase,j,'aa12  ', aa(1,2)
        write(*,900) icase,j,'aa13  ', aa(1,3)
        write(*,900) icase,j,'aa21  ', aa(2,1)
        write(*,900) icase,j,'aa22  ', aa(2,2)
        write(*,900) icase,j,'aa23  ', aa(2,3)
        write(*,900) icase,j,'aa31  ', aa(3,1)
        write(*,900) icase,j,'aa32  ', aa(3,2)
        write(*,900) icase,j,'aa33  ', aa(3,3)
        write(*,900) icase,j,'bb11  ', bb(1,1)
        write(*,900) icase,j,'bb12  ', bb(1,2)
        write(*,900) icase,j,'bb13  ', bb(1,3)
        write(*,900) icase,j,'bb21  ', bb(2,1)
        write(*,900) icase,j,'bb22  ', bb(2,2)
        write(*,900) icase,j,'bb23  ', bb(2,3)
        write(*,900) icase,j,'bb31  ', bb(3,1)
        write(*,900) icase,j,'bb32  ', bb(3,2)
        write(*,900) icase,j,'bb33  ', bb(3,3)

 400  continue
 500  continue

 900  format(I3,',',I3,',',A6,',',E26.18)
      stop
      end
