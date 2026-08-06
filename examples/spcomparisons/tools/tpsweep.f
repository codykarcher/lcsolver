      program tpsweep
c------------------------------------------------------------------
c     Sweeps trefftz1 exactly as cdsum.f:386-476 sets it up.
c     stdin lines: S bref bw bs bo zw lams lamt rcls rclt fLo
c                  CL CLh bh boh zh lamh
c     stdout: sefftp CLtp CDtp   (one line per input line)
c------------------------------------------------------------------
      implicit real(a-h,l-z)
      integer nsurf, ktip
      integer npout(2), npinn(2), npimg(2)
      real b(2), bs(2), bo(2), bop(2), zcent(2)
      real po(2), gammat(2), gammas(2)
      real CLsurfsp(2), CLsurf(2)
      logical Lspec
      parameter (idim=500)
      real yc(idim), ycp(idim), zc(idim), zcp(idim)
      real gc(idim), vc(idim), wc(idim), vnc(idim)
      integer ifrst(2), ilast(2)

 10   read(*,*,end=99) Sref, bref, bw, bsw, bow, zw,
     &                 rlams, rlamt, rcls, rclt, fLo,
     &                 CL, CLh, bh, boh, zh, rlamh

      Lspec = .true.
      b(1)  = bw
      bs(1) = bsw
      bo(1) = bow
      bop(1) = bow * 0.2
      zcent(1) = zw
      gammas(1) = rlams*rcls
      gammat(1) = rlamt*rclt
      po(1) = 1.0
      CLsurfsp(1) = CL - CLh

      b(2) = bh
      bs(2) = boh
      bo(2) = boh
      bop(2) = boh
      zcent(2) = zh
      gammas(2) = 1.0
      gammat(2) = rlamh
      po(2) = 1.0
      CLsurfsp(2) = CLh

      nsurf = 2
      npout(1) = 20
      npinn(1) = 6
      npimg(1) = 3
      npout(2) = 10
      npinn(2) = 0
      if(bo(2) .eq. 0.0) then
       npimg(2) = 0
      else
       npimg(2) = 2
      endif
      ktip = 16

      call trefftz1(nsurf, npout, npinn, npimg,
     &             Sref, bref,
     &             b,bs,bo,bop, zcent,
     &             po,gammat,gammas, fLo, ktip,
     &             Lspec,CLsurfsp,
     &             CLsurf,CLtp,CDtp,sefftp,
     &             idim,ifrst,ilast,
     &             yc,zc,gc,vc,wc,vnc, ycp,zcp)

      write(*,'(3(1x,e14.7))') sefftp, CLtp, CDtp
      go to 10
 99   continue
      end
