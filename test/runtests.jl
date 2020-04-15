using ACSF
using Test, LinearAlgebra, BenchmarkTools
using JuLIP, JuLIP.Testing
using ACSF: acsf

@testset "ACSF.jl" begin
    include("test.jl")
end

