using TASOPT, Printf, NLsolve
T = TASOPT; C = TASOPT.CryoTank; S = TASOPT.structures
open("/tmp/insul_ref.csv","w") do io
    println(io,"case,R,Wfuel,boiloff,t1,t2,Tfuel,z,TSL,M,xftank,qfac,hvap,dt")
    for (nm,R,Wf,bo) in [("a",1.9,3000*9.81,0.2),("b",1.9,3000*9.81,0.5),("c",2.2,5000*9.81,0.3)]
        cs = S.SingleBubble(radius=R)
        layout = S.FuselageLayout{S.SingleBubble}(cross_section=cs)
        fuse = S.Fuselage{S.SingleBubble}(layout=layout)
        ft = T.fuselage_tank()
        ft.Wfuelintank = Wf; ft.rhofuel = 70.8; ft.rhofuelgas = 1.33
        ft.ullage_frac = 0.05; ft.pvent = 2.0e5; ft.clearance_fuse = 0.1
        ft.ARtank = 2.0; ft.ew = 0.9; ft.ftankadd = 0.1; ft.theta_inner = 1.2
        ft.Tfuel = 20.4; ft.qfac = 1.0; ft.hvap = 446.0e3
        ft.boiloff_rate = bo; ft.iinsuldes = [1,2]
        ft.t_insul = [0.05, 0.10]
        ft.inner_material = T.StructuralAlloy("Al-2219-T87")
        ft.material_insul = [T.ThermalInsulator("polyurethane27"), T.ThermalInsulator("polyurethane32")]
        z = 11000.0; TSL = 288.2; M = 0.8; xf = 20.0
        _,_,Taw = C.freestream_heat_coeff(z, TSL, M, xf)
        dT = Taw - ft.Tfuel
        f(x) = C.res_MLI_thick(x, fuse, ft, z, TSL, M, xf, 40)
        guess = zeros(4); guess[1]=0.0; guess[2]=ft.Tfuel+1.0
        guess[3] = ft.Tfuel + dT*0.05/0.15; guess[4] = ft.Tfuel + dT - 1.0
        sol = nlsolve(f, guess, ftol=1e-7)
        @printf(io,"%s,%.17g,%.17g,%.17g,0.05,0.10,20.4,%.17g,%.17g,%.17g,%.17g,1.0,446000.0,%.17g\n",
                nm,R,Wf,bo,z,TSL,M,xf,sol.zero[1])
    end
end
println("ok")
