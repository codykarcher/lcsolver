# Dump the v3 compressor maps (pyCycle-derived) and their extrapolated grids.
using TASOPT, Interpolations
E = TASOPT.engine
const OUT = "/Users/codykarcher/Dropbox/research/edi/examples/tasopt/julia_ref"

function dumpmap(io, name, m)
    d = m.defaults
    println(io, "$name,defaults,Nc,$(d.Nc)")
    println(io, "$name,defaults,Rline,$(d.Rline)")
    println(io, "$name,defaults,Wc,$(d.Wc)")
    println(io, "$name,defaults,PR,$(d.PR)")
    println(io, "$name,defaults,polyeff,$(d.polyeff)")
    println(io, "$name,NcMap,-," * join(m.NcMap, " "))
    println(io, "$name,RlineMap,-," * join(m.RlineMap, " "))
    for (lbl, A) in [("WcMap", m.WcMap), ("PRMap", m.PRMap),
                     ("polyeff_Map", m.polyeffMap)]
        for i in 1:size(A,1)
            println(io, "$name,$lbl,$i," * join(A[i,:], " "))
        end
    end
end

open(joinpath(OUT,"maps_raw.csv"),"w") do io
    println(io,"map,field,row,values")
    dumpmap(io, "Fan", E.FanMap)
    dumpmap(io, "LPC", E.LPCMap)
    dumpmap(io, "HPC", E.HPCMap)
end

# Interpolant samples, including inside the extrapolation padding.
open(joinpath(OUT,"maps_interp.csv"),"w") do io
    println(io,"map,N,R,Wc,PR,polyeff,dWc_dN,dWc_dR,dPR_dN,dPR_dR,dpe_dN,dpe_dR")
    for (nm, m) in [("Fan",E.FanMap),("LPC",E.LPCMap),("HPC",E.HPCMap)]
        for N in [0.15, 0.35, 0.62, 0.805, 1.0, 1.12, 1.4],
            R in [0.5, 1.1, 1.75, 2.2, 2.63, 3.0, 3.5]
            gW = Interpolations.gradient(m.itp_Wc, N, R)
            gP = Interpolations.gradient(m.itp_PR, N, R)
            gE = Interpolations.gradient(m.itp_polyeff, N, R)
            println(io, join([nm,N,R,m.itp_Wc(N,R),m.itp_PR(N,R),m.itp_polyeff(N,R),
                              gW[1],gW[2],gP[1],gP[2],gE[1],gE[2]], ","))
        end
    end
end

# calculate_compressor_speed_and_efficiency at assorted operating points.
open(joinpath(OUT,"maps_speed_eff.csv"),"w") do io
    println(io,"map,pratio,mb,piD,mbD,NbD,epol0,Nb,epol,dNb_dpi,dNb_dmb,depol_dpi,depol_dmb,N,R")
    for (nm,m) in [("Fan",E.FanMap),("LPC",E.LPCMap),("HPC",E.HPCMap)]
        piD = nm=="Fan" ? 1.68 : (nm=="LPC" ? 3.0 : 10.0)
        for pr in [piD*0.6, piD*0.85, piD, piD*1.1], mb in [0.55, 0.8, 1.0, 1.15]
            try
                r = E.calculate_compressor_speed_and_efficiency(m, pr, mb, piD, 1.0, 1.0, 0.9)
                println(io, join([nm,pr,mb,piD,1.0,1.0,0.9, r...], ","))
            catch e
                println(io, join([nm,pr,mb,piD,1.0,1.0,0.9,"NaN","NaN","NaN","NaN","NaN","NaN","NaN","NaN"], ","))
            end
        end
    end
end
println("done")
