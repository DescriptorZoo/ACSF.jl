module ACSF

using JuLIP
using LinearAlgebra

export acsf, acsf_desc, G2, G4, cutoff_func, get_max_Rc, generate_sf_parameters, set_Behler2011

unit_BOHR = 0.52917721067
acsfPI = 3.14159265358979

function generate_sf_parameters(;n=5, cutoff=6.0, bohr=false)
    if bohr
        cutoff = cutoff/unit_BOHR
    end
    m = collect(0:n)
    n_pow = n.^(m./n)
    eta_m = (n_pow./cutoff).^2
    R_s = cutoff./n_pow
    eta_s = zeros(n+1)
    for mi=1:n+1
        eta_s[mi] = 1. / (R_s[n - mi - 1] - R_s[n - mi])^2
    end
    # !!!!!! FLIPPED eta_s ARRAY HERE !!!! REVERSED ORDER NOT MENTIONED IN PAPER !!!!!!
    # ALSO R_s and eta_s shifted !!! NOT MENTIONED AGAIN !!!
    return eta_m, R_s[1:n+1], eta_s[end:-1:1]
end

function set_Behler2011()
    cutoff = 6.5
    Gpack = [
        [   9.0,  100.0,  200.0,  350.0,  600.0, 1000.0, 2000.0, 4000.0], #G2_etas
        [   1.0,    1.0,    1.0,    1.0,   30.0,   30.0,   30.0,   30.0,  
           80.0,   80.0,   80.0,   80.0,  150.0,  150.0,  150.0,  150.0, 
          150.0,  150.0,  150.0,  150.0,  250.0,  250.0,  250.0,  250.0, 
          250.0,  250.0,  250.0,  250.0,  450.0,  450.0,  450.0,  450.0, 
          450.0,  450.0,  450.0,  450.0,  800.0,  800.0,  800.0,  800.0, 
          800.0,  800.0,  800.0], #G4_etas
        [  -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,  
           -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,  
           -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,  
           -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,  
           -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,   -1.0,    1.0,
           -1.0,    1.0,    1.0], #G4_lambdas
        [   1.0,    1.0,    2.0,    2.0,    1.0,    1.0,    2.0,    2.0,
            1.0,    1.0,    2.0,    2.0,    1.0,    1.0,    2.0,    2.0,
            4.0,    4.0,   16.0,   16.0,    1.0,    1.0,    2.0,    2.0,   
            4.0,    4.0,   16.0,   16.0,    1.0,    1.0,    2.0,    2.0,
            4.0,    4.0,   16.0,   16.0,    1.0,    1.0,    2.0,    2.0,
            4.0,    4.0,   16.0] #G4_zetas
    ]
    Gpack[1] ./= 10000.0 * unit_BOHR^2
    Gpack[2] ./= 10000.0 * unit_BOHR^2
    return Dict("G2" => [cutoff,Gpack[1]], 
                "G4" => [cutoff,Gpack[2],Gpack[3],Gpack[4]])
end

function get_max_Rc(Gparams)    
    Rc = 0.0
    if "G2" in keys(Gparams)
        if Gparams["G2"][1] > Rc
            Rc = Gparams["G2"][1]
        end
    end
    if "G4" in keys(Gparams)
        if Gparams["G4"][1] > Rc
            Rc = Gparams["G4"][1]
        end
    end
    return Rc
end

function cutoff_Tanh(dij, Rc)
    fc = zeros(size(dij))
    dij_in_Rc = dij .<= Rc
    fc[dij_in_Rc] = (tanh.(1. .- dij[dij_in_Rc]./Rc)) .^ 3.0
    return fc  
end

function cutoff_Cosine(dij, Rc)
    fc = zeros(size(dij))
    dij_in_Rc = dij .<= Rc
    fc[dij_in_Rc] = 0.5 .* (cos.(dij[dij_in_Rc] .* π/Rc) .+ 1.)
    return fc  
end

cutoff_func(dij, Rc; cutTanh=false) = cutTanh ? cutoff_Tanh(dij, Rc) : cutoff_Cosine(dij, Rc)

function G2(dj, Rc, eta; Rs=nothing, cutTanh=false)
    fc_ij = cutoff_func(dj, Rc, cutTanh=cutTanh)
    if Rs == nothing
        return [sum(exp.(-eta[p] .* (dj .^ 2)) .* 
                fc_ij) for p = 1:length(eta)]
    else
        return [sum(exp.(-eta[p] .* (dj .- Rs[p]) .^ 2.0) .* 
                fc_ij) for p = 1:length(eta)]
    end
end

function G4(Rj, dj, Rc, eta, lambd, zeta; cutTanh=false)
    counts = collect(1:size(dj,1))
    # So far no other way to do this simpler.
    Rij = collect(Iterators.flatten([[Rj[j] for k in counts] for j in counts]))
    Rik = collect(Iterators.flatten([[Rj[k] for k in counts] for j in counts]))
    Dij = collect(Iterators.flatten([[dj[j] for k in counts] for j in counts]))
    Dik = collect(Iterators.flatten([[dj[k] for k in counts] for j in counts]))
    Djk = norm.(Rik .- Rij)
    ###
    cos_theta_ijk = [Rik[i,:]' * Rij[i,:] for i = 1:size(Rij,1)] ./ (Dij .* Dik)
    exp_arg = Dij.^2 .+ Dik.^2 .+ Djk.^2
    fc_all = cutoff_func(Dij, Rc, cutTanh=cutTanh) .* 
             cutoff_func(Dik, Rc, cutTanh=cutTanh) .*  
             cutoff_func(Djk, Rc, cutTanh=cutTanh)
    jkmask = collect(Iterators.flatten([[k>j for k in counts] for j in counts]))
    return [ 2^(1 - zeta[p]) * 
            sum( ((1 .+ lambd[p] .* cos_theta_ijk[jkmask]) .^ zeta[p]) .*
                 exp.(-eta[p] .* exp_arg[jkmask]) .* fc_all[jkmask]
               ) for p = 1:length(eta) ]
end

function G9(Rj, dj, Rc, eta, lambd, zeta; cutTanh=false, l=nothing)
    if l == nothing
        l=1
    end
    if l<1
        l=1
    end
    counts = collect(1:size(dj,1))
    # So far no other way to do this simpler.
    Rij = collect(Iterators.flatten([[Rj[j] for k in counts] for j in counts]))
    Rik = collect(Iterators.flatten([[Rj[k] for k in counts] for j in counts]))
    Dij = collect(Iterators.flatten([[dj[j] for k in counts] for j in counts]))
    Dik = collect(Iterators.flatten([[dj[k] for k in counts] for j in counts]))
    Djk = norm.(Rik .- Rij)
    ###
    cos_theta_ijk = [Rik[i,:]' * Rij[i,:] for i = 1:size(Rij,1)] ./ (Dij .* Dik)
    exp_arg = Dij.^2 .+ Dik.^2 .+ Djk.^2
    fc_all = cutoff_func(Dij, Rc, cutTanh=cutTanh) .* 
             cutoff_func(Dik, Rc, cutTanh=cutTanh) .*  
             cutoff_func(Djk, Rc, cutTanh=cutTanh)
    jkmask = collect(Iterators.flatten([[k>j for k in counts] for j in counts]))
    return [ 
            sum( ((sqrt.(Complex.(1.0 .- cos_theta_ijk[jkmask] .^ 2)) .^ (2 * l * zeta[p])) .* 
                   exp.(1im * 2 * l * zeta[p])) .^ 2 .*
                 exp.(-eta[p] .* exp_arg[jkmask]) .* fc_all[jkmask]
               ) for p = 1:length(eta) ]
            #sum( abs2.((sqrt.(Complex.(1.0 .- cos_theta_ijk[jkmask].^2)) .^ (2 * l * zeta[p])) .* 
            #       exp.(1im * 2 * l * zeta[p])) .*
            #     exp.(-eta[p] .* exp_arg[jkmask]) .* fc_all[jkmask]
            #   ) for p = 1:length(eta) ]
end

function acsf_desc(Rs; Gparams=nothing, cutfunc=nothing, useG9=false, l=nothing)
    ds = norm.(Rs)
    descriptor = []
    if cutfunc == "Tanh" || cutfunc == "tanh"
        cutTanh = true
    else 
        cutTanh = false
    end
    if Gparams == nothing
        Gparams = set_Behler2011()
    end
    if "G2" in keys(Gparams)
        # Concatenate radial basis functions
        if length(Gparams["G2"]) > 2
            descriptor = vcat(descriptor,G2(ds, Gparams["G2"][1],
                                                Gparams["G2"][2],
                                                Rs=Gparams["G2"][3],
                                                cutTanh=cutTanh))
        else
            descriptor = vcat(descriptor,G2(ds, Gparams["G2"][1],
                                                Gparams["G2"][2],
                                                cutTanh=cutTanh))
        end
    end
    if "G4" in keys(Gparams)
        # Concatenate angular basis functions
        if useG9
            descriptor = vcat(descriptor,G9(Rs, ds, Gparams["G4"][1],
                                                    Gparams["G4"][2],
                                                    Gparams["G4"][3],
                                                    Gparams["G4"][4],
                                                    cutTanh=cutTanh, l=l))
        else
            descriptor = vcat(descriptor,G4(Rs, ds, Gparams["G4"][1],
                                                    Gparams["G4"][2],
                                                    Gparams["G4"][3],
                                                    Gparams["G4"][4],
                                                    cutTanh=cutTanh))
        end
    end
    return descriptor
end

function acsf(at; Gparams=nothing, cutfunc=nothing, useG9=false, l=nothing)
    if Gparams == nothing
        Gparams = set_Behler2011()
    end
    representation = []
    ni = []
    nj = []
    nR = []
    nd = []
    for (i, j, R) in pairs(neighbourlist(at, get_max_Rc(Gparams)))
        push!(ni, i)
        push!(nj, j)
        push!(nR, R)
        push!(nd, norm(R))
    end
    for i = 1:length(at)
        Rs = nR[(ni .== i)]
        push!(representation,acsf_desc(Rs, Gparams=Gparams, cutfunc=cutfunc, useG9=useG9, l=l))
    end
    return representation
end

end # module
