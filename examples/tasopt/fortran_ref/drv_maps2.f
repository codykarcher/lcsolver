      program drvm2
      implicit real (a-h,o-z)
      real mb, Nb, mflux, Mguess
      real alpha(6), beta(6)
      integer i, j, nair
      include 'tfmap.inc'
      include 'airfrac.inc'
      nair = 5
      open(9,file='/tmp/maps2_ref.csv',status='unknown')
c---- etmap: turbine efficiency, spanning both sides of the design point
      do 10 i = 1, 6
        dh = -2.6e5 - 3.0e4*float(i)
        mb =  8.0 + 1.0*float(i)
        Nb =  0.9 + 0.05*float(i)
        Tt = 1400.0 + 20.0*float(i)
        cpt = 1250.0
        Rt = 288.0
        call etmap(dh,mb,Nb, 4.0,11.0,1.05, 0.889, Tmaph,
     &             Tt,cpt,Rt, ept,
     &             e_dh,e_mb,e_Nb,e_Tt,e_cpt,e_Rt)
        write(9,'(a,i3,a,e24.16)') 'etmap,',i,',',ept
 10   continue
c---- gas_mass: static state at specified mass flux
      to = 800.0
      call gassum(alpha,nair, to, so,dsdt, ho,dhdt, cpo, ro)
      po = 4.0e5
      do 20 j = 1, 5
        mflux = 120.0 + 40.0*float(j)
        Mguess = 0.5
        call gas_mass(alpha,nair, po,to,ho,so,cpo,ro, mflux,Mguess,
     &                p,t,h,s,cp,r)
        write(9,'(a,i3,a,3(e24.16,a),e24.16)') 'gas_mass,',j,',',
     &        p,',',t,',',h,',',s
 20   continue
      close(9)
      end
