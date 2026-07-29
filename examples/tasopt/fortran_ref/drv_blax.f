      program drv_blax
c---- Axisymmetric BL + wake over a 737-like fuselage, driven by the real
c---- potential-flow solution from axisol -- so the geometry, the arc-length
c---- spacing and uinv are all the ones fusebl would hand it, not invented.
c----
c---- Cases:
c----   1  M = 0.80, Re/l = 1.0e7, smooth wall     (the 737-ish case)
c----   2  M = 0.80, Re/l = 1.0e7, fexcr = 1.03    (excrescence factor on)
c----   3  M = 0.20, Re/l = 3.0e6                  (low speed, lower Re)
c----   4  M = 0.80, Re/l = 1.0e7, edge-type tail  (ifclose = 1, dy .ne. 0)
c----   5  M = 0.80, Re/l = 4.0e7, blunter body    (thicker, higher Re)
c----
c---- gfortran -fdefault-real-8 -O0 -o drv_blax drv_blax.f blax.f blsys.f
c----          axisol.f gaussn.f
      implicit real (a-h,m,o-z)
      integer nldim, nl, ilte, nc, iclose, icase, i, ibl
      parameter (nldim = 60)
      real xb(nldim), zb(nldim), sb(nldim), dyb(nldim), qb(nldim)
      real bb(nldim), rnb(nldim)
      real ueb(nldim), dsb(nldim), thb(nldim), tsb(nldim), dcb(nldim)
      real cfb(nldim), cdb(nldim), ctb(nldim), hkb(nldim), phb(nldim)

      pi = 3.14159265358979323846264338327950

c---- The inputs are dumped alongside the outputs so the test can drive blax
c---- with the Fortran's own geometry and uinv, and thus measure blax alone
c---- rather than blax composed with axisol.
      write(*,'(A)')
     &  'case,i,s,b,rn,uinv,ue,ds,th,ts,dc,cf,cd,ct,hk,ph'

      do 500 icase = 1, 5

        xnose = 0.0
        xend  = 37.8
        xblend1 = 6.1
        xblend2 = 30.0
        Sfuse = 12.0
        anose = 1.8
        btail = 1.6
        iclose = 0
        rMach = 0.80
        Reunit = 1.0e7
        fex = 1.0
        nc = 30

        if(icase.eq.2) fex = 1.03
        if(icase.eq.3) then
         rMach  = 0.20
         Reunit = 3.0e6
        endif
        if(icase.eq.4) iclose = 1
        if(icase.eq.5) then
         Reunit = 4.0e7
         Sfuse  = 16.0
         anose  = 2.2
         btail  = 2.0
        endif

        call axisol(xnose,xend,xblend1,xblend2, Sfuse,
     &              anose,btail,iclose, rMach, nc,
     &              nldim, nl, ilte, xb,zb,sb,dyb,qb)

c------ perimeter, exactly as fusebl builds it
        if(iclose.eq.0) then
         do ibl = 1, nl
           bb(ibl) = 2.0*pi*zb(ibl)
         enddo
        else
         do ibl = 1, nl
           bb(ibl) = 2.0*pi*zb(ibl) + 4.0*dyb(ibl)
         enddo
        endif

c------ dr/dn, exactly as fusebl builds it
        do ibl = 2, nl-1
          dxm = xb(ibl) - xb(ibl-1)
          dzm = zb(ibl) - zb(ibl-1)
          dsm = sb(ibl) - sb(ibl-1)
          dxp = xb(ibl+1) - xb(ibl)
          dzp = zb(ibl+1) - zb(ibl)
          dsp = sb(ibl+1) - sb(ibl)
          dxo = (dxm/dsm)*dsp + (dxp/dsp)*dsm
          dzo = (dzm/dsm)*dsp + (dzp/dsp)*dsm
          rnb(ibl) = dxo / sqrt(dxo**2 + dzo**2)
        enddo
        rnb(1) = rnb(2)
     &         - (0.5*(sb(3)+sb(2)) - sb(1))
     &          *(rnb(3)-rnb(2))/(sb(3)-sb(2))
        rnb(nl) = rnb(nl-1)
     &         + (sb(nl) - 0.5*(sb(nl-1)+sb(nl-2)))
     &          *(rnb(nl-1)-rnb(nl-2))/(sb(nl-1)-sb(nl-2))
        rnb(1) = max( rnb(1) , 0.0 )
        rnb(nl) = max( rnb(nl) , 0.0 )

c------ blax reads cd(1) without ever writing it; in the program it is a
c------ zeroed COMMON block, so make that explicit here.
        do ibl = 1, nldim
          ueb(ibl) = 0.
          dsb(ibl) = 0.
          thb(ibl) = 0.
          tsb(ibl) = 0.
          dcb(ibl) = 0.
          cfb(ibl) = 0.
          cdb(ibl) = 0.
          ctb(ibl) = 0.
          hkb(ibl) = 0.
          phb(ibl) = 0.
        enddo

        call blax(nldim, nl,ilte,
     &            sb, bb, rnb, qb, Reunit, rMach, fex,
     &            ueb, dsb, thb, tsb, dcb, cfb, cdb, ctb, hkb, phb )

        write(*,901) icase, 0, float(nl), float(ilte),
     &               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     &               0.0, 0.0
        do i = 1, nl
          write(*,901) icase, i, sb(i), bb(i), rnb(i), qb(i),
     &                 ueb(i), dsb(i), thb(i), tsb(i), dcb(i),
     &                 cfb(i), cdb(i), ctb(i), hkb(i), phb(i)
        enddo

 500  continue
 901  format(I2,',',I4,',',13(E26.18,','),E26.18)
      stop
      end
