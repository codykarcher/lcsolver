using TASOPT, Printf
C = TASOPT.CryoTank
# Exercise the interpolations directly on synthetic mission tables, so the
# reference values are the reference's arithmetic rather than a whole model.
times = [0.0, 600.0, 1800.0, 5400.0, 12000.0, 18000.0, 21600.0]
mdots = [0.30, 0.28, 0.24, 0.18, 0.12, 0.02, 0.0]
mdots0 = [0.30, 0.28, 0.00, 0.18, 0.12, 0.02, 0.0]   # a zero mid-mission
Qs    = [3400.0, 3300.0, 3000.0, 2600.0, 2550.0, 2900.0, 3450.0]
open("/tmp/tanktools_ref.csv","w") do io
    println(io,"kind,t,val")
    for t in [0.0, 300.0, 600.0, 1200.0, 5400.0, 9000.0, 15000.0, 21600.0]
        # Fuel flow: exponential interpolation, as find_mdot_time does.
        function mdot_of(t, ms)
            for i in 1:length(times)
                if times[i] == t; return ms[i]; end
            end
            for i in 1:(length(times)-1)
                if times[i] <= t < times[i+1]
                    t0, tf = times[i], times[i+1]
                    m0, mf = ms[i], ms[i+1]
                    if m0 > 0
                        k = log(mf/m0)/(tf-t0)
                        return m0*exp(k*(t-t0))
                    else
                        return 0.0
                    end
                end
            end
        end
        function Q_of(t)
            for i in 1:length(times)
                if times[i] == t; return Qs[i]; end
            end
            for i in 1:(length(times)-1)
                if times[i] <= t < times[i+1]
                    t0, tf = times[i], times[i+1]
                    return Qs[i] + (Qs[i+1]-Qs[i])/(tf-t0)*(t-t0)
                end
            end
        end
        @printf(io,"mdot,%.17g,%.17g\n", t, mdot_of(t, mdots))
        @printf(io,"mdot0,%.17g,%.17g\n", t, mdot_of(t, mdots0))
        @printf(io,"Q,%.17g,%.17g\n", t, Q_of(t))
    end
end
println("ok")
