using TakagiFactorization
using LinearAlgebra
using Test

@testset "Takagi" begin
    @testset "Examples" begin
        A₁ = convert(Matrix{Complex{Float64}}, [1.0 2.0; 2.0 1.0])
        d₁, U₁ = takagi_factor(A₁)
        @test A₁ ≈ transpose(U₁) * d₁ * U₁
        @test d₁ ≈ Diagonal([3.0, 1.0])
        @test_broken U₁ ≈ [1 1; 1im -1im] / √2
        A₂ = convert(Matrix{Complex{Float64}}, [0.0 1.0; 1.0 0.0])
        d₂, U₂ = takagi_factor(A₂)
        @test A₂ ≈ transpose(U₂) * d₂ * U₂
        @test d₂ ≈ Diagonal([1.0, 1.0])
        @test U₂ ≈ [1 1; -1im 1im] / √2
    end
    @testset "BigFloat" begin
        A₁ = convert(Matrix{Complex{BigFloat}}, [1.0 2.0; 2.0 1.0])
        d₁, U₁ = takagi_factor(A₁)
        @test A₁ ≈ transpose(U₁) * d₁ * U₁
        @test d₁ ≈ Diagonal([3.0, 1.0])
        @test_broken U₁ ≈ [1 1; 1im -1im] / √big(2)
        A₂ = convert(Matrix{Complex{BigFloat}}, [0.0 1.0; 1.0 0.0])
        d₂, U₂ = takagi_factor(A₂)
        @test A₂ ≈ transpose(U₂) * d₂ * U₂
        @test d₂ ≈ Diagonal([1.0, 1.0])
        @test U₂ ≈ [1 1; -1im 1im] / √big(2)
    end
end
