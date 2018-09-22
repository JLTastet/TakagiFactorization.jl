using TakagiFactorization
using LinearAlgebra
using Test

@testset "Takagi" begin
    @testset "Examples" begin
        A₁ = convert(Matrix{Complex{Float64}}, [1.0 2.0; 2.0 1.0])
        d₁, U₁ = takagi_factor(A₁)
        @test A₁ ≈ transpose(U₁) * d₁ * U₁ atol=2eps(Float64)*sum(abs.(A₁))
        @test d₁ ≈ Diagonal([3.0, 1.0]) atol=eps(Float64)*sum(d₁)
        @test U₁ ≈ [1 1; -1im 1im] / √2 atol=2eps(Float64)*sum(abs.(U₁))
        A₂ = convert(Matrix{Complex{Float64}}, [0.0 1.0; 1.0 0.0])
        d₂, U₂ = takagi_factor(A₂)
        @test A₂ ≈ transpose(U₂) * d₂ * U₂ atol=2eps(Float64)*sum(abs.(A₂))
        @test d₂ ≈ Diagonal([1.0, 1.0]) atol=eps(Float64)*sum(d₂)
        @test U₂ ≈ [1 1; -1im 1im] / √2 atol=2eps(Float64)*sum(abs.(U₂))
    end
    @testset "BigFloat" begin
        A₁ = convert(Matrix{Complex{BigFloat}}, [1.0 2.0; 2.0 1.0])
        d₁, U₁ = takagi_factor(A₁)
        @test A₁ ≈ transpose(U₁) * d₁ * U₁ atol=2eps(BigFloat)*sum(abs.(A₁))
        @test d₁ ≈ Diagonal([3.0, 1.0]) atol=eps(BigFloat)*sum(d₁)
        @test U₁ ≈ [1 1; -1im 1im] / √big(2) atol=2eps(BigFloat)*sum(abs.(U₁))
        A₂ = convert(Matrix{Complex{BigFloat}}, [0.0 1.0; 1.0 0.0])
        d₂, U₂ = takagi_factor(A₂)
        @test A₂ ≈ transpose(U₂) * d₂ * U₂ atol=2*eps(BigFloat)*sum(abs.(A₂))
        @test d₂ ≈ Diagonal([1.0, 1.0]) atol=eps(BigFloat)*sum(d₂)
        @test U₂ ≈ [1 1; -1im 1im] / √big(2) atol=2eps(BigFloat)*sum(abs.(U₂))
    end
    @testset "Sorting" begin
        @testset "Ascending" begin
            A₁ = convert(Matrix{Complex{Float64}}, [1.0 2.0; 2.0 1.0])
            d₁, U₁ = takagi_factor(A₁, sort=+1)
            @test A₁ ≈ transpose(U₁) * d₁ * U₁ atol=2eps(Float64)*sum(abs.(A₁))
            @test d₁ ≈ Diagonal([1.0, 3.0]) atol=eps(Float64)*sum(d₁)
            @test U₁ ≈ [-1im 1im; 1 1] / √2 atol=2eps(Float64)*sum(abs.(U₁))
            A₂ = convert(Matrix{Complex{Float64}}, [0.0 1.0; 1.0 0.0])
            d₂, U₂ = takagi_factor(A₂, sort=+1)
            @test A₂ ≈ transpose(U₂) * d₂ * U₂ atol=2eps(Float64)*sum(abs.(A₂))
            @test d₂ ≈ Diagonal([1.0, 1.0]) atol=eps(Float64)*sum(d₂)
            @test U₂ ≈ [1 1; -1im 1im] / √2 atol=2eps(Float64)*sum(abs.(U₂))
        end
        @testset "Descending" begin
            A₁ = convert(Matrix{Complex{Float64}}, [1.0 2.0; 2.0 1.0])
            d₁, U₁ = takagi_factor(A₁, sort=-1)
            @test A₁ ≈ transpose(U₁) * d₁ * U₁ atol=2eps(Float64)*sum(abs.(A₁))
            @test d₁ ≈ Diagonal([3.0, 1.0]) atol=eps(Float64)*sum(d₁)
            @test U₁ ≈ [1 1; -1im 1im] / √2 atol=2eps(Float64)*sum(abs.(U₁))
            A₂ = convert(Matrix{Complex{Float64}}, [0.0 1.0; 1.0 0.0])
            d₂, U₂ = takagi_factor(A₂, sort=-1)
            @test A₂ ≈ transpose(U₂) * d₂ * U₂ atol=2eps(Float64)*sum(abs.(A₂))
            @test d₂ ≈ Diagonal([1.0, 1.0]) atol=eps(Float64)*sum(d₂)
            @test U₂ ≈ [1 1; -1im 1im] / √2 atol=2eps(Float64)*sum(abs.(U₂))
        end
    end
    @testset "Exceptions" begin
        A₁ = convert(Matrix{Complex{Float64}}, [1 2; 3 4])
        @test_throws ArgumentError takagi_factor(A₁)
        A₂ = convert(Matrix{Complex{Float64}}, [1 2 3; 3 5 6])
        @test_throws ArgumentError takagi_factor(A₂)
        A₃ = convert(Matrix{Complex{Float64}}, [1 2; 2 1])
        U₃_good = zeros(Complex{Float64}, 2, 2)
        U₃_bad  = zeros(Complex{Float64}, 2, 3)
        d₃_good = zeros(Float64, 2)
        d₃_bad  = zeros(Float64, 3)
        @test_throws ArgumentError takagi_factor!(A₃, d₃_good, U₃_bad)
        @test_throws ArgumentError takagi_factor!(A₃, d₃_bad, U₃_good)
    end
end
