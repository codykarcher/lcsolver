using TASOPT, Printf
T = TASOPT; C = TASOPT.CryoTank
open("/tmp/inner_tank_ref.csv","w") do io
    println(io,"case,Rfuse,dRfuse,wfb,nwebs,Wfuel,rhof,rhog,ullage,pvent,clear,AR,ew,ftankadd,theta,tins1,tins2,Wtank,Winsul,Vfuel,Rtank_outer,l_tank,l_cyl")
    cases = [
      ("single", 1.9, 0.0, 0.0, 0, 5.0e5, 70.8, 1.33, 0.05, 2.0e5, 0.1, 2.0, 0.9, 0.1, 1.2, [0.05,0.10]),
      ("bubble", 1.9, 0.38, 0.6, 1, 5.0e5, 70.8, 1.33, 0.05, 2.0e5, 0.1, 2.0, 0.9, 0.1, 1.2, [0.05,0.10]),
      ("bigger", 2.2, 0.0, 0.0, 0, 1.2e6, 70.8, 1.33, 0.08, 3.0e5, 0.15, 2.4, 0.85, 0.15, 1.0, [0.08,0.12]),
    ]
    for (nm,R,dR,wfb,nw,Wf,rl,rg,ull,pv,cl,AR,ew,fa,th,tins) in cases
        S = T.structures
        if nw == 0
            cs = S.SingleBubble(radius=R, bubble_lower_downward_shift=dR)
            layout = S.FuselageLayout{S.SingleBubble}(cross_section = cs)
            fuse = S.Fuselage{S.SingleBubble}(layout = layout)
        else
            cs = S.MultiBubble(radius=R, bubble_lower_downward_shift=dR,
                               bubble_center_y_offset=wfb, n_webs=nw)
            layout = S.FuselageLayout{S.MultiBubble}(cross_section = cs)
            fuse = S.Fuselage{S.MultiBubble}(layout = layout)
        end
        ft = T.fuselage_tank()
        ft.Wfuelintank = Wf; ft.rhofuel = rl; ft.rhofuelgas = rg
        ft.ullage_frac = ull; ft.pvent = pv; ft.clearance_fuse = cl
        ft.ARtank = AR; ft.ew = ew; ft.ftankadd = fa; ft.theta_inner = th
        ft.inner_material = T.StructuralAlloy("Al-2219-T87")
        ft.material_insul = [T.ThermalInsulator("polyurethane27"), T.ThermalInsulator("polyurethane32")]
        r = C.size_inner_tank(fuse, ft, tins)
        @printf(io,"%s,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                nm,R,dR,wfb,nw,Wf,rl,rg,ull,pv,cl,AR,ew,fa,th,tins[1],tins[2],
                r.Wtank, r.Winsul_sum, r.Vfuel, r.Rtank_outer, r.l_tank, r.l_cyl)
    end
end
println("ok")
