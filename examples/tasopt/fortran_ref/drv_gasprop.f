      program drv_gasprop
      implicit real (a-z)
      integer nh, nc, nn, no
      write(*,'(A)') 'name,value'
      ttf = 300.0d0
      tt3 = 700.0d0
      tt4 = 1400.0d0
      rfuel = 519.65d0
      cpfuel = 2240.0d0
      hsfuel = -4.675d6
      nh = 2
      nc = 1
      nn = 0
      no = 0
      call gasprop(ttf,tt3,tt4,rfuel,cpfuel,hsfuel,
     &             nh,nc,nn,no, cp3,gam3, cp4,gam4, f)
      write(*,900) 'f   ',f
      write(*,900) 'cp3 ',cp3
      write(*,900) 'gam3',gam3
      write(*,900) 'cp4 ',cp4
      write(*,900) 'gam4',gam4
 900  format(A4,',',E24.16)
      stop
      end
