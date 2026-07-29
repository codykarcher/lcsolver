using TASOPT, Printf
T = TASOPT; C = TASOPT.CryoTank; S = TASOPT.structures
open("/tmp/opt_outer_ref.csv","w") do io
    println(io,"case,Rfuse,Winner,l_cyl,clear,AR,ftankadd,th1,th2,N0,Ninterm,Wtank_opt")
    for (nm,R,Wi,lc,N0) in [("a",1.9,3.0e5,6.0,1.0),("b",1.9,8.0e5,12.0,1.0),("c",2.2,5.0e5,9.0,4.0)]
        cs = S.SingleBubble(radius=R)
        layout = S.FuselageLayout{S.SingleBubble}(cross_section = cs)
        fuse = S.Fuselage{S.SingleBubble}(layout = layout)
        ft = T.fuselage_tank()
        ft.clearance_fuse = 0.1; ft.ARtank = 2.0; ft.ftankadd = 0.1
        ft.theta_outer = [1.0, 2.2]; ft.Ninterm = N0
        ft.inner_material = T.StructuralAlloy("Al-2219-T87")
        N = C.optimize_outer_tank(fuse, ft, Wi, lc)
        W = C.size_outer_tank(fuse, ft, Wi, lc, N).Wtank
        @printf(io,"%s,%.17g,%.17g,%.17g,0.1,2.0,0.1,1.0,2.2,%.17g,%.17g,%.17g\n",nm,R,Wi,lc,N0,N,W)
    end
end
println("ok")
