module TakagiFactorization

export takagi_factor!

function takagi_factor!(
    A :: AbstractArray{Complex{T}, 2},
    d :: AbstractArray{T, 1},
    U :: AbstractArray{Complex{T}, 2};
    sort = 0,
    maxsweeps = 50
) where {T <: AbstractFloat}

    n = size(A, 1)
    if size(A, 2) ≠ n
        throw(ValueError("A must be a square matrix"))
    end
    if size(U) ≠ size(A)
        throw(ValueError("U must be a square matrix with the same dimensions as A"))
    end
    if size(d, 1) ≠ n
        throw(ValueError("d must have length n for a n×n matrix A"))
    end
    red = T(0.04) / n^4
    ev = zeros(Complex{T}, n, 2)

    for p in 1:n
        ev[p,1] = zero(T)
        ev[p,2] = A[p,p]
    end

    fill!(U, zero(T))
    for p in 1:n
        U[p,p] = one(T)
    end

    done = false
    nsweeps = 0
    while !done && (nsweeps += 1) ≤ maxsweeps
        thresh = sum(abs2(A[p,q]) for q in 2:n for p in 1:q)
        if thresh ≤ sym_eps(T)
            done = true
            continue
        end

        thresh = (nsweeps < 4) ? thresh*red : zero(T)

        for q in 2:n
            for p in 1:q
                off = abs2(A[p,q])
                sqp = abs2(ev[p,2])
                sqq = abs2(ev[q,2])
                if nsweeps > 4 && off < sym_eps(T)*(sqp+sqq)
                    A[p,q] = zero(T)
                elseif off > thresh
                    t = abs(sqp-sqq) / 2
                    f = if t > eps(T)
                        sign(sqp-sqq) * (ev[q,2]*A[p,q]' + ev[p,2]'*A[p,q])
                    else
                        (sqp == 0) ? one(T) : √(ev[q,2]/ev[p,2])
                    end
                    t += √(t^2 + abs2(f))
                    f /= t

                    ev[p,2] = A[p,p] + (ev[p,1] += A[p,q]*f')
                    ev[q,2] = A[q,q] + (ev[q,1] -= A[p,q]*f )

                    t = abs2(f)
                    c⁻¹ = √(t + 1)
                    f /= c⁻¹
                    t /= c⁻¹*(c⁻¹+1)

                    for j in 1:p
                        x = A[j,p]
                        y = A[j,q]
                        A[j,p] = x + (f'*y - t*x)
                        A[j,q] = y - (f*x + t*y)
                    end

                    for j in p+2:q
                        x = A[p,j]
                        y = A[j,q]
                        A[p,j] = x + (f'*y - t*x)
                        A[j,q] = y - (f*x + t*y)
                    end

                    for j in q+2:n
                        x = A[p,j]
                        y = A[q,j]
                        A[p,j] = x + (f'*y - t*x)
                        A[q,j] = y - (f*x + t*y)
                    end

                    A[p,q] = zero(T)

                    for j in 1:n
                        x = U[j,p]
                        y = U[j,q]
                        U[j,p] = x + (f*y - t*x)
                        U[j,q] = y - (f'*x + t*y)
                    end
                end # elseif off > thresh
            end # for p in 1:q
        end # for q in 2:n

        for p in 1:n
            ev[p,1] = zero(T)
            A[p,p] = ev[p,2]
        end
    end # for nsweeps in 1:maxsweeps

    if !done
        @warn "Bad convergence in takagi_factor!"
    else
        for p in 1:n
            # d[p] = abs(A[p,p])
            # if d[p] > eps(T) && d[p] ≠ real(A[p,p])
            #     U[:,p] .*= √(A[p,p]/d[p])
            # end
            App = A[p,p]
            d[p] = abs(App)
            if d[p] > eps(T) && d[p] ≠ real(App)
                f = √(App/d[p])
                for q in 1:n
                    U[q,p] *= f
                end
            end
        end

        if sort ≠ 0
            for p in 1:n-1
                j = p
                t = d[p]
                for q in p+2:n
                    if sort*(t-d[q]) > 0
                        t = d[j=q]
                    end
                    if j == p
                        continue
                    end
                    d[j] = d[p]
                    d[p] = t
                    for q in 1:n
                        x = U[q,p]
                        U[q,p] = U[q,j]
                        U[q,j] = x
                    end
                end # for q in p+2:n
            end # for p in 1:n-1
        end # if sort ≠ 0
    end
end

sq_eps(x)  = 4*eps(x)^2
sym_eps(x) = 2*eps(x)^2

end # module
