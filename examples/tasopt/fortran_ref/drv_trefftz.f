      program drv_trefftz
c---- Trefftz-plane induced drag for a wing alone and for wing + tail,
c---- with and without specified surface lift.
      implicit real(a-h,l-z)
      integer nsurf, ktip, idim, icase, isurf, i
      parameter (idim = 360)
      integer npout(2), npinn(2), npimg(2), ifrst(2), ilast(2)
      real b(2), bs(2), bo(2), bop(2), zcent(2)
      real po(2), gammat(2), gammas(2)
      real CLsurfsp(2), CLsurf(2)
      real yc(idim), zc(idim), gc(idim), vc(idim), wc(idim), vnc(idim)
      real ycp(idim), zcp(idim)
      logical Lspec

      write(*,'(A)') 'case,name,value'
      do 500 icase = 1, 4

      Sref = 124.0
      bref = 35.0
      fLo = -0.3
      ktip = 16

c---- wing
      nsurf = 1
      npout(1) = 20
      npinn(1) = 6
      npimg(1) = 3
      b(1)   = 35.0
      bs(1)  = 10.5
      bo(1)  = 3.6
      bop(1) = 3.6
      zcent(1) = 0.0
      po(1) = 1.0
      gammat(1) = 0.28
      gammas(1) = 0.78
      CLsurfsp(1) = 0.55
      Lspec = .false.

      if (icase .eq. 2) then
        Lspec = .true.
      endif

      if (icase .ge. 3) then
c------ add a horizontal tail above the wing plane
        nsurf = 2
        npout(2) = 12
        npinn(2) = 4
        npimg(2) = 2
        b(2)   = 13.0
        bs(2)  = 4.0
        bo(2)  = 2.0
        bop(2) = 2.0
        zcent(2) = 2.5
        po(2) = -0.18
        gammat(2) = 0.30
        gammas(2) = 0.80
        CLsurfsp(2) = -0.08
        if (icase .eq. 4) Lspec = .true.
      endif

      call trefftz1(nsurf, npout, npinn, npimg,
     &              Sref, bref,
     &              b,bs,bo,bop, zcent,
     &              po,gammat,gammas, fLo,ktip,
     &              Lspec,CLsurfsp,
     &              CLsurf,CL,CD,spanef,
     &              idim,ifrst,ilast,
     &              yc,zc,gc,vc,wc,vnc, ycp,zcp)

      write(*,900) icase,'CL   ',CL
      write(*,900) icase,'CD   ',CD
      write(*,900) icase,'spane',spanef
      do isurf = 1, nsurf
        write(*,901) icase,'CLs',isurf,CLsurf(isurf)
      enddo
      do i = 1, ilast(nsurf), 7
        write(*,901) icase,'gc ',i,gc(i)
        write(*,901) icase,'vnc',i,vnc(i)
        write(*,901) icase,'wc ',i,wc(i)
      enddo

 500  continue
 900  format(I2,',',A5,',',E24.16)
 901  format(I2,',',A3,I3.3,',',E24.16)
      stop
      end
