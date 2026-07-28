      program drv_surfw
      implicit real (a-z)
      integer iwplan, icase
      write(*,'(A)') 'case,name,value'
      do icase = 1, 3
        gee = 9.81d0
        po = 120000.0d0
        b = 35.0d0
        bs = 12.0d0
        bo = 3.6d0
        co = 6.0d0
        zs = 5.0d0
        lambdat = 0.25d0
        lambdas = 0.65d0
        gammat = 0.22d0
        gammas = 0.70d0
        Nload = 3.0d0
        iwplan = 1
        We = 30000.0d0
        Winn = 12000.0d0
        Wout = 8000.0d0
        dyWinn = 40000.0d0
        dyWout = 60000.0d0
        sweep = 26.0d0
        wbox = 0.50d0
        hboxo = 0.1268d0
        hboxs = 0.1266d0
        rh = 0.75d0
        fLt = -0.05d0
        tauweb = 1.38d8
        sigcap = 2.06d8
        sigstrut = 2.06d8
        Ecap = 6.9d10
        Eweb = 6.9d10
        Gcap = 2.6d10
        Gweb = 2.6d10
        rhoweb = 2700.0d0
        rhocap = 2700.0d0
        rhostrut = 2700.0d0
        rhofuel = 817.0d0
        if (icase .eq. 2) then
          iwplan = 0
          We = 0.0d0
        endif
        if (icase .eq. 3) then
          iwplan = 2
        endif
        call surfw(gee,po,b,bs,bo,co,zs,
     &       lambdat,lambdas,gammat,gammas,
     &       Nload,iwplan,We,
     &       Winn,Wout,dyWinn,dyWout,
     &       sweep,wbox,hboxo,hboxs,rh, fLt,
     &       tauweb,sigcap,sigstrut,Ecap,Eweb,Gcap,Gweb,
     &       rhoweb,rhocap,rhostrut,rhofuel,
     &       Ss,Ms,tbwebs,tbcaps,EIcs,EIns,GJs,
     &       So,Mo,tbwebo,tbcapo,EIco,EIno,GJo,
     &       Astrut,lsp,cosLs,
     &       Wscen,Wsinn,Wsout,dxWsinn,dxWsout,dyWsinn,dyWsout,
     &       Wfcen,Wfinn,Wfout,dxWfinn,dxWfout,dyWfinn,dyWfout,
     &       Wweb,Wcap,Wstrut, dxWweb,dxWcap,dxWstrut)
        write(*,'(I2,A,A,A,E24.16)') icase,',','Ss',',',Ss
        write(*,'(I2,A,A,A,E24.16)') icase,',','Ms',',',Ms
        write(*,'(I2,A,A,A,E24.16)') icase,',','tbwebs',',',tbwebs
        write(*,'(I2,A,A,A,E24.16)') icase,',','tbcaps',',',tbcaps
        write(*,'(I2,A,A,A,E24.16)') icase,',','EIcs',',',EIcs
        write(*,'(I2,A,A,A,E24.16)') icase,',','EIns',',',EIns
        write(*,'(I2,A,A,A,E24.16)') icase,',','GJs',',',GJs
        write(*,'(I2,A,A,A,E24.16)') icase,',','So',',',So
        write(*,'(I2,A,A,A,E24.16)') icase,',','Mo',',',Mo
        write(*,'(I2,A,A,A,E24.16)') icase,',','tbwebo',',',tbwebo
        write(*,'(I2,A,A,A,E24.16)') icase,',','tbcapo',',',tbcapo
        write(*,'(I2,A,A,A,E24.16)') icase,',','EIco',',',EIco
        write(*,'(I2,A,A,A,E24.16)') icase,',','EIno',',',EIno
        write(*,'(I2,A,A,A,E24.16)') icase,',','GJo',',',GJo
        write(*,'(I2,A,A,A,E24.16)') icase,',','Astrut',',',Astrut
        write(*,'(I2,A,A,A,E24.16)') icase,',','lsp',',',lsp
        write(*,'(I2,A,A,A,E24.16)') icase,',','cosLs',',',cosLs
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wscen',',',Wscen
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wsinn',',',Wsinn
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wsout',',',Wsout
        write(*,'(I2,A,A,A,E24.16)') icase,',','dxWsinn',',',dxWsinn
        write(*,'(I2,A,A,A,E24.16)') icase,',','dxWsout',',',dxWsout
        write(*,'(I2,A,A,A,E24.16)') icase,',','dyWsinn',',',dyWsinn
        write(*,'(I2,A,A,A,E24.16)') icase,',','dyWsout',',',dyWsout
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wfcen',',',Wfcen
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wfinn',',',Wfinn
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wfout',',',Wfout
        write(*,'(I2,A,A,A,E24.16)') icase,',','dxWfinn',',',dxWfinn
        write(*,'(I2,A,A,A,E24.16)') icase,',','dxWfout',',',dxWfout
        write(*,'(I2,A,A,A,E24.16)') icase,',','dyWfinn',',',dyWfinn
        write(*,'(I2,A,A,A,E24.16)') icase,',','dyWfout',',',dyWfout
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wweb',',',Wweb
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wcap',',',Wcap
        write(*,'(I2,A,A,A,E24.16)') icase,',','Wstrut',',',Wstrut
        write(*,'(I2,A,A,A,E24.16)') icase,',','dxWweb',',',dxWweb
        write(*,'(I2,A,A,A,E24.16)') icase,',','dxWcap',',',dxWcap
        write(*,'(I2,A,A,A,E24.16)') icase,',','dxWstrut',',',dxWstrut
      enddo
      end
