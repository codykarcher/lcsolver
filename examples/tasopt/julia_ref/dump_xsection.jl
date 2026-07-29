using TASOPT, Printf
T = TASOPT
open("/tmp/xsection_ref.csv","w") do io
    println(io, "kind,R,dR,wfb,nwebs,Rscale,perim,area")
    for (R, dR) in [(1.9,0.0),(1.9,0.4),(2.5,0.8)]
        cs = T.SingleBubble(radius=R, bubble_lower_downward_shift=dR)
        for Rs in [R, 0.8R, 0.5R, 1.2R]
            p, a = T.scaled_cross_section(cs, Rs)
            @printf(io,"single,%.17g,%.17g,0,0,%.17g,%.17g,%.17g\n",R,dR,Rs,p,a)
        end
    end
    for (R, dR, wfb, n) in [(1.9,0.0,0.6,1),(1.9,0.38,0.6,1),(2.2,0.5,0.9,2)]
        cs = T.MultiBubble(radius=R, bubble_lower_downward_shift=dR,
                           bubble_center_y_offset=wfb, n_webs=n)
        for Rs in [R, 0.8R, 0.5R, 1.2R]
            p, a = T.scaled_cross_section(cs, Rs)
            @printf(io,"multi,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g\n",R,dR,wfb,n,Rs,p,a)
        end
    end
end
println("ok")
