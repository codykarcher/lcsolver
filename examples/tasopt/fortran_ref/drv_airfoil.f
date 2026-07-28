      program drv_airfoil
c---- Airfoil database read + tri-cubic evaluation, over a grid of
c---- (Mach, cl, tau) inside the table and outside it in both directions.
      implicit real (a-h,o-z)
      integer idim,jdim,kdim,ldim, i,j,k
      parameter (idim=60, jdim=20, kdim=20, ldim=3)
      integer nAMa,nAcl,nAtau,nAfun
      real AMa(idim), Acl(jdim), Atau(kdim), ARe
      real          A(idim,jdim,kdim,ldim),
     &            A_M(idim,jdim,kdim,ldim),
     &           A_cl(idim,jdim,kdim,ldim),
     &          A_tau(idim,jdim,kdim,ldim),
     &         A_M_cl(idim,jdim,kdim,ldim),
     &        A_M_tau(idim,jdim,kdim,ldim),
     &       A_cl_tau(idim,jdim,kdim,ldim),
     &     A_M_cl_tau(idim,jdim,kdim,ldim)
      real cdf, cdp, cdw, cm
      character*80 fname

      fname = '/Users/codykarcher/Desktop/Tasopt2.16/air/C.air'
      call airtable(fname, idim,jdim,kdim,ldim,
     &  nAMa,nAcl,nAtau,nAfun, AMa,Acl,Atau,ARe,
     &  A,A_M,A_cl,A_tau,A_M_cl,A_M_tau,A_cl_tau,A_M_cl_tau)

      write(*,'(A)') 'name,value'
      write(*,'(A,E24.16)') 'ARe,', ARe
c---- a few spline derivative entries, to check airtable itself
      write(*,'(A,E24.16)') 'A_M_1,',      A_M(5,3,4,1)
      write(*,'(A,E24.16)') 'A_cl_1,',     A_cl(5,3,4,1)
      write(*,'(A,E24.16)') 'A_tau_1,',    A_tau(5,3,4,2)
      write(*,'(A,E24.16)') 'A_M_cl_1,',   A_M_cl(9,4,3,2)
      write(*,'(A,E24.16)') 'A_M_tau_1,',  A_M_tau(9,4,3,3)
      write(*,'(A,E24.16)') 'A_cl_tau_1,', A_cl_tau(9,4,3,1)
      write(*,'(A,E24.16)') 'A_Mclt_1,',   A_M_cl_tau(12,5,5,2)

      do 10 i = 1, 5
      do 20 j = 1, 4
      do 30 k = 1, 4
        rMach = 0.30 + 0.13*float(i)
        cl    = 0.35 + 0.13*float(j)
        tau   = 0.085 + 0.018*float(k)
        call airfun(cl,tau,rMach, idim,jdim,kdim,ldim,
     &    nAMa,nAcl,nAtau,nAfun, AMa,Acl,Atau,ARe,
     &    A,A_M,A_cl,A_tau,A_M_cl,A_M_tau,A_cl_tau,A_M_cl_tau,
     &    cdf,cdp,cdw,cm)
        write(*,900) 'f',i,j,k,cdf,cdp,cdw,cm
 30   continue
 20   continue
 10   continue
 900  format(A1,3I2.2,',',3(E24.16,','),E24.16)
      stop
      end
