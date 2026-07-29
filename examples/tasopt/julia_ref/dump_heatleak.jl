using TASOPT, Printf
T = TASOPT; C = TASOPT.CryoTank; S = TASOPT.structures
open("/tmp/heatleak_ref.csv","w") do io
    println(io,"case,R,dR,wfb,nw,l_cyl,l_inner,Rinner,t1,t2,Tfuel,z,TSL,M,xftank,ifuel,qfac,Sh1,Sh2,Sh3,Q")
    cases = [("cruise",1.9,0.0,0.0,0,5.0,7.0,1.65,0.05,0.10,20.4,11000.0,288.2,0.8,20.0,40,1.3),
             ("ground",1.9,0.0,0.0,0,5.0,7.0,1.65,0.05,0.10,20.4,0.0,303.2,0.0,20.0,40,1.3),
             ("bubble",1.9,0.38,0.6,1,5.0,7.0,1.65,0.05,0.10,20.4,11000.0,288.2,0.8,20.0,40,1.3),
             ("methane",1.9,0.0,0.0,0,5.0,7.0,1.65,0.05,0.10,111.5,11000.0,288.2,0.8,20.0,11,1.3),
             ("vacuum",1.9,0.0,0.0,0,5.0,7.0,1.65,0.05,0.10,20.4,11000.0,288.2,0.8,20.0,40,1.3)]
    for (nm,R,dR,wfb,nw,lc,li,Ri,t1,t2,Tf,z,TSL,M,xf,ifu,qf) in cases
        if nw == 0
            cs = S.SingleBubble(radius=R, bubble_lower_downward_shift=dR)
            layout = S.FuselageLayout{S.SingleBubble}(cross_section=cs)
            fuse = S.Fuselage{S.SingleBubble}(layout=layout)
        else
            cs = S.MultiBubble(radius=R, bubble_lower_downward_shift=dR,
                               bubble_center_y_offset=wfb, n_webs=nw)
            layout = S.FuselageLayout{S.MultiBubble}(cross_section=cs)
            fuse = S.Fuselage{S.MultiBubble}(layout=layout)
        end
        # Head areas at the three interfaces, as size_inner_tank would give.
        Sh = Float64[]
        for r in [Ri, Ri+t1, Ri+t1+t2]
            _, A = T.structures.scaled_cross_section(cs, r)
            push!(Sh, 2*A*(0.333 + 0.667*(1.0/2.0)^1.6)^0.625)
        end
        ft = T.fuselage_tank()
        ft.qfac = qf; ft.t_insul = [t1,t2]; ft.Tfuel = Tf
        ft.l_cyl_inner = lc; ft.l_inner = li; ft.Rinnertank = Ri
        ft.Shead_insul = Sh
        ft.material_insul = nm == "vacuum" ?
            [T.ThermalInsulator("vacuum"), T.ThermalInsulator("polyurethane32")] :
            [T.ThermalInsulator("polyurethane27"), T.ThermalInsulator("polyurethane32")]
        Q = C.tankWthermal(fuse, ft, z, TSL, M, xf, ifu)
        @printf(io,"%s,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                nm,R,dR,wfb,nw,lc,li,Ri,t1,t2,Tf,z,TSL,M,xf,ifu,qf,Sh[1],Sh[2],Sh[3],Q)
    end
end
println("ok")
