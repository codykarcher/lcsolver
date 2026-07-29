using TASOPT, Printf
T = TASOPT; C = TASOPT.CryoTank; S = TASOPT.structures
open("/tmp/outer_tank_ref.csv","w") do io
    println(io,"case,Rfuse,dRfuse,wfb,nwebs,Winner,l_cyl,Ninterm,clear,AR,ftankadd,th1,th2,Wtank,Wcyl,Whead,Wstiff,Souter,Shead,Scyl,t_cyl,t_head,l_outer")
    cases = [
      ("single", 1.9, 0.0, 0.0, 0, 3.0e5, 6.0, 0.0, 0.1, 2.0, 0.1, 1.0, 2.2),
      ("bubble", 1.9, 0.38, 0.6, 1, 3.0e5, 6.0, 2.0, 0.1, 2.0, 0.1, 1.0, 2.2),
      ("many",   2.2, 0.0, 0.0, 0, 8.0e5, 12.0, 6.0, 0.15, 2.4, 0.15, 0.6, 2.6),
    ]
    for (nm,R,dR,wfb,nw,Wi,lc,Nint,cl,AR,fa,t1,t2) in cases
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
        ft.clearance_fuse = cl; ft.ARtank = AR; ft.ftankadd = fa
        ft.theta_outer = [t1, t2]
        ft.inner_material = T.StructuralAlloy("Al-2219-T87")
        r = C.size_outer_tank(fuse, ft, Wi, lc, Nint)
        @printf(io,"%s,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                nm,R,dR,wfb,nw,Wi,lc,Nint,cl,AR,fa,t1,t2,
                r.Wtank,r.Wcyl,r.Whead,r.Wstiff,r.Souter,r.Shead,r.Scyl,r.t_cyl,r.t_head,r.l_outer)
    end
end
println("ok")
