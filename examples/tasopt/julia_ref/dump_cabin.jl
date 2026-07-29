using TASOPT, Printf
S = TASOPT.structures
open("/tmp/cabin_ref.csv","w") do io
    println(io,"kind,a,b,c,d,e,r1,r2,r3")
    for w in [2.0, 3.0, 3.5, 4.0, 5.0, 5.5, 6.5, 7.5]
        n = S.findSeatsAbreast(w)
        @printf(io,"abreast,%.17g,0,0,0,0,%d,0,0\n", w, n)
        ys = S.arrange_seats(n, w)
        @printf(io,"seaty,%.17g,0,0,0,0,%.17g,%.17g,%.17g\n", w, ys[1], ys[end], sum(ys))
    end
    for pax in [100, 180, 250, 400], w in [3.5, 5.5]
        lc, xs, n = S.place_cabin_seats(pax, w)
        @printf(io,"place,%d,%.17g,0,0,0,%.17g,%d,%d\n", pax, w, lc, n, length(xs))
    end
    for R in [1.9, 2.2], wfb in [0.0, 0.6], nfw in [0, 1], th in [-0.1, 0.0, 0.2]
        w = S.find_cabin_width(R, wfb, nfw, th, 0.5)
        @printf(io,"width,%.17g,%.17g,%d,%.17g,0.5,%.17g,0,0\n", R, wfb, nfw, th, w)
    end
    for R in [1.9, 2.2], hs in [0.4, 0.5, 0.6]
        t1 = S.find_floor_angles(false, R, 0.0; h_seat = hs)
        @printf(io,"angle1,%.17g,%.17g,0,0,0,%.17g,0,0\n", R, hs, t1)
    end
    for R in [2.6, 3.0], t1 in [-0.2, 0.0], df in [2.0, 2.4]
        a, b = S.find_floor_angles(true, R, 0.0; θ1 = t1, d_floor = df)
        @printf(io,"angle2,%.17g,%.17g,%.17g,0,0,%.17g,%.17g,0\n", R, t1, df, a, b)
    end
end
println("ok")
