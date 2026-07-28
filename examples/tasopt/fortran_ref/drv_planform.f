      program drv_planform
      implicit real (a-h,l-n,o-z)
      integer i
      write(*,'(A)') 'name,v1,v2,v3,v4,v5'
      do 10 i = 1, 4
        S = 25.0 + 4.0*float(i)
        AR = 4.0 + 0.6*float(i)
        rlam = 0.20 + 0.05*float(i)
        qne = 12000.0 + 900.0*float(i)
        CLmax = 0.6 + 0.1*float(i)
        call tailpo(S,AR,rlam,qne,CLmax, b,co,po)
        write(*,900) 'tailpo',i,b,co,po
 10   continue
      do 20 i = 1, 4
        b = 33.0 + 1.5*float(i)
        bs = 9.0 + 0.6*float(i)
        bo = 3.2 + 0.15*float(i)
        rlamt = 0.15 + 0.02*float(i)
        rlams = 0.65 + 0.03*float(i)
        sweep = 22.0 + 2.0*float(i)
        call surfdx(b,bs,bo,rlamt,rlams,sweep, dx,macco)
        write(*,901) 'surfdx',i,dx,macco
 20   continue
      do 30 i = 1, 4
        W = 600000.0 + 40000.0*float(i)
        CL = 0.50 + 0.03*float(i)
        qinf = 11000.0 + 500.0*float(i)
        AR = 8.0 + 0.5*float(i)
        etasi = 0.25 + 0.02*float(i)
        bo = 3.2 + 0.15*float(i)
        rlamt = 0.15 + 0.02*float(i)
        rlams = 0.65 + 0.03*float(i)
        call wingsc(W,CL,qinf,AR,etasi,bo,rlamt,rlams, S,b,bs,co)
        write(*,902) 'wingsc',i,S,b,bs,co
        bfix = b
        bsfix = bs
        call wingAc(W,CL,qinf,ARo,etasi,bo,rlamt,rlams,
     &              S2,bfix,bsfix,co2)
        write(*,902) 'wingAc',i,S2,ARo,bsfix,co2
 30   continue
 900  format(A6,I2.2,',',2(E24.16,','),E24.16)
 901  format(A6,I2.2,',',E24.16,',',E24.16)
 902  format(A6,I2.2,',',3(E24.16,','),E24.16)
      stop
      end
