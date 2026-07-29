      program drv_blclosure
      implicit real (a-h,m,o-z)
      integer i, j, k
      write(*,'(A)') 'name,i,j,value'
      do 10 i = 1, 8
      do 20 j = 1, 5
        hk = 1.2 + 0.6*float(i)
        rt = 10.0**(1.0 + 0.8*float(j))
        msq = 0.04*float(j)
        call hkin(hk, msq, v, d1, d2)
        write(*,900) 'hkin ',i,j,v
        call hsl(hk, rt, msq, v, d1, d2, d3)
        write(*,900) 'hsl  ',i,j,v
        call hst(hk, rt, msq, v, d1, d2, d3)
        write(*,900) 'hst  ',i,j,v
        call cfl(hk, rt, msq, v, d1, d2, d3)
        write(*,900) 'cfl  ',i,j,v
        call cft(hk, rt, msq, v, d1, d2, d3)
        write(*,900) 'cft  ',i,j,v
        call dil(hk, rt, v, d1, d2)
        write(*,900) 'dil  ',i,j,v
        call dilw(hk, rt, v, d1, d2)
        write(*,900) 'dilw ',i,j,v
        call hct(hk, msq, v, d1, d2)
        write(*,900) 'hct  ',i,j,v
        call dit(1.7+0.1*float(i), 0.1*float(j), 0.002, 0.01*float(j),
     &           v, d1, d2, d3, d4)
        write(*,900) 'dit  ',i,j,v
 20   continue
 10   continue
 900  format(A5,',',I3,',',I3,',',E24.16)
      stop
      end
