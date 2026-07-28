      program drv_gascalc
C     Exercise the gascalc.f mixture routines on dry air + C14H30 fuel.
C     NOTE: every actual argument is a named variable. Passing literal
C     constants through these implicit-interface calls under -O produced
C     garbage for the pressure-ratio argument.
      implicit real (a-z)
      integer i, n, ifuel
      real alpha(5), beta(5), gamma(5), lambda(5)
      n = 5
      ifuel = 24
      alpha(1) = 0.7532d0
      alpha(2) = 0.2314d0
      alpha(3) = 0.0006d0
      alpha(4) = 0.0020d0
      alpha(5) = 0.0128d0
      do i = 1, 5
        beta(i) = 0.0d0
      enddo
      beta(5) = 1.0d0
      eff  = 0.90d0
      tgss = 400.0d0
      tair = 700.0d0
      tfue = 300.0d0
      write(*,'(A)') 'case,k,v'

      do i = 0, 12
        tt = 250.0d0 + dble(i)*150.0d0
        call gassum(alpha,n, tt, sx,stx, hx,htx, cpx,rx)
        write(*,'(A,I3,6(A,E24.16))') 'gassum,',i,',',sx,',',stx,',',
     &        hx,',',htx,',',cpx,',',rx
      enddo

      call gasfuel(ifuel,gamma,n)
      write(*,'(A,I3,5(A,E24.16))') 'gasfuel,',0,',',gamma(1),',',
     &      gamma(2),',',gamma(3),',',gamma(4),',',gamma(5)

      do i = 0, 8
        tt = 300.0d0 + dble(i)*200.0d0
        call gassum(alpha,n, tt, sx,stx, hx,htx, cpx,rx)
        call gas_tset(alpha,n, hx, tgss, tout)
        write(*,'(A,I3,2(A,E24.16))') 'tset,',i,',',tt,',',tout
      enddo

      to = 288.2d0
      po = 1.0132d5
      call gassum(alpha,n, to, so,sto, ho,hto, cpo,ro)

      do i = 1, 10
        prr = 1.0d0 + dble(i)*3.0d0
        call gas_prat(alpha,n, po,to,ho,so,cpo,ro, prr, eff,
     &                p,t,h,s,cp,r)
        write(*,'(A,I3,4(A,E24.16))') 'prat,',i,',',p,',',t,',',h,',',s
      enddo

      do i = 1, 8
        dh = dble(i)*5.0d4
        call gas_delh(alpha,n, po,to,ho,so,cpo,ro, dh, eff,
     &                p,t,h,s,cp,r)
        write(*,'(A,I3,4(A,E24.16))') 'delh,',i,',',p,',',t,',',h,',',s
      enddo

      do i = 1, 8
        tb = 1000.0d0 + dble(i)*100.0d0
        call gas_burn(alpha,beta,gamma,n,ifuel, tair,tfue,tb,
     &                f,lambda)
        write(*,'(A,I3,6(A,E24.16))') 'burn,',i,',',f,',',lambda(1),
     &        ',',lambda(2),',',lambda(3),',',lambda(4),',',lambda(5)
      enddo
      end
