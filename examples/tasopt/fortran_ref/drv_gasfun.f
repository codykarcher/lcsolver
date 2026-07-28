      program drv_gasfun
C     Emit gasfun() over a temperature sweep for every implemented gas.
      implicit real (a-z)
      integer i, k, ng, igases(11)
      data igases / 1,2,3,4,5,11,12,13,14,18,24 /
      ng = 11
      write(*,'(A)') 'igas,t,s,s_t,h,h_t,cp,r'
      do k = 1, ng
        do i = 0, 30
          t = 200.0d0 + dble(i)*90.0d0
          call gasfun(igases(k), t, s,s_t, h,h_t, cp,r)
          write(*,'(I3,A,F10.3,6(A,E24.16))') igases(k),',',t,',',
     &          s,',',s_t,',',h,',',h_t,',',cp,',',r
        enddo
      enddo
      end
