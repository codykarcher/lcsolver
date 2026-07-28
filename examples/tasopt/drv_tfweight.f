      program drv_tfweight
      implicit real (a-h,l-z)
      integer iengwgt, i, k
      include 'constants.inc'
      pi = 3.1415926535897932384626
      gee = 9.81
      lb_N = 1.0 / 4.44822
      write(*,'(A)') 'name,Weng,Wnac,Webare,Snace1'
      do 10 iengwgt = 0, 2
      do 20 k = 1, 2
      do 30 i = 1, 3
        Gearf = 1.0
        if (k .eq. 2) Gearf = 3.0
        OPR = 30.0 + 6.0*float(i)
        BPR = 5.0 + 2.0*float(i)
        rmdotc = 40.0 + 6.0*float(i)
        dfan = 1.6 + 0.2*float(i)
        rSnace = 14.0
        dlcomp = 0.9 + 0.1*float(i)
        eng = 2.0
        feadd = 0.10
        fpylon = 0.10
        call tfweight(iengwgt,Gearf,OPR,BPR,rmdotc,dfan,rSnace,
     &                dlcomp,eng,feadd,fpylon,
     &                Weng,Wnac,Webare,Snace1)
        write(*,900) iengwgt,k,i,Weng,Wnac,Webare,Snace1
 30   continue
 20   continue
 10   continue
 900  format('w',3I2.2,',',3(E24.16,','),E24.16)
      stop
      end
