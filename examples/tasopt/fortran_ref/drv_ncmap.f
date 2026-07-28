      program drvn
      implicit real (a-h,o-z)
      real mb, Nb, Nb_pi, Nb_mb, pi, piD
      integer ic, ip, im
      include 'tfmap.inc'
      open(9,file='/tmp/ncmap_ref.csv',status='unknown')
      do 10 ic = 1, 3
      do 20 ip = 1, 4
      do 30 im = 1, 4
        mb = 34.0 + 3.0*float(im)
        if(ic.eq.1) then
          piD = 1.685
          pi = piD + 0.06*float(ip-2)
          call Ncmap(pi,mb, piD,40.0,1.05, Cmapf, Nb,Nb_pi,Nb_mb)
        elseif(ic.eq.2) then
          piD = 1.935
          pi = piD + 0.06*float(ip-2)
          call Ncmap(pi,mb, piD,40.0,1.05, Cmapl, Nb,Nb_pi,Nb_mb)
        else
          piD = 2.100
          pi = piD + 0.06*float(ip-2)
          call Ncmap(pi,mb, piD,40.0,1.05, Cmaph, Nb,Nb_pi,Nb_mb)
        endif
        write(9,'(3(i3,a),2(e24.16,a),e24.16)') ic,',',ip,',',im,',',
     &        Nb,',',Nb_pi,',',Nb_mb
 30   continue
 20   continue
 10   continue
      close(9)
      end
