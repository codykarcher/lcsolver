using TASOPT, Printf
C = TASOPT.CryoTank
open("/tmp/stiffeners_ref.csv","w") do io
    println(io, "kind,a1,a2,phimax,kmax")
    for th in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
        p, k = C.stiffeners_bendingM(th)
        @printf(io, "inner,%.17g,0,%.17g,%.17g\n", th, p, k)
    end
    for th1 in [0.3, 0.6, 1.0, 1.4], th2 in [1.6, 2.0, 2.4, 2.8]
        p, k = C.stiffeners_bendingM_outer(th1, th2)
        @printf(io, "outer,%.17g,%.17g,%.17g,%.17g\n", th1, th2, p, k)
    end
end
open("/tmp/k1head_ref.csv","w") do io
    println(io, "AR,K1")
    for ar in [1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0, 2.25, 2.6, 2.9, 3.0]
        @printf(io, "%.17g,%.17g\n", ar, C.find_K1_head(ar))
    end
end
open("/tmp/stiffw_ref.csv","w") do io
    println(io, "kind,W,Rtank,perim,s_a,rho,th1,th2,Nstiff,l_cyl,E,Wstiff")
    for (kind, th1, th2, N, l, E) in [("inner",1.2,0.0,2.0,0.0,0.0),
                                      ("inner",0.8,0.0,2.0,0.0,0.0),
                                      ("outer",1.0,2.2,2.0,5.0,7.3e10),
                                      ("outer",0.6,2.6,6.0,12.0,7.3e10)]
        for W in [1e5, 5e5], R in [1.5, 2.2]
            s_a = 1.7e8; rho = 2825.0
            perim = 2*pi*R
            Ws = C.stiffener_weight(kind, W, R, perim, s_a, rho, th1, th2, N, l, E)
            @printf(io, "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    kind, W, R, perim, s_a, rho, th1, th2, N, l, E, Ws)
        end
    end
end
println("ok")
