@testset "Grids" begin

    # with a shift to the grid element, check if produce the correct floored index
    function check(grid, range, shift, idx_shift)
        for i in range
            @test(floor(grid, grid[i] + shift) == i + idx_shift)
            # if floor(grid, grid[i] + shift) != i + idx_shift
            #     return false
            # end
        end
        return true
    end

    @testset "UniformGrid" begin
        uniform = Grid.Uniform{Float64,4}(0.0, 1.0, (true, true))
        @test floor(uniform, 0.0) == 1
        @test floor(uniform, uniform[1]) == 1

        δ = 1.0e-12
        check(uniform, 2:uniform.size - 1, δ, 0)
        check(uniform, 2:uniform.size - 1, -δ, -1)

        @test floor(uniform, uniform[end]) == uniform.size - 1
        @test floor(uniform, 1.0) == uniform.size - 1

        uniform = Grid.Uniform{Float64,2}(0.0, 1.0, (true, true))
        @test floor(uniform, 0.0) == 1
        @test floor(uniform, uniform[1]) == 1

        δ = 1.0e-12
        check(uniform, 2:uniform.size - 1, δ, 0)
        check(uniform, 2:uniform.size - 1, -δ, -1)

        @test floor(uniform, uniform[end]) == uniform.size - 1
        @test floor(uniform, 1.0) == uniform.size - 1
    end


    @testset "Log Grid for Tau" begin
        β = 10.0
        tau = Grid.tau(β, 0.2β, 8)
        @test floor(tau, 0.0) == 1
        @test floor(tau, tau[1]) == 1 # tau[1]=0^+ is special

        δ = 1.0e-12
        check(tau, 2:tau.size - 1, δ, 0)
        check(tau, 2:tau.size - 1, -δ, -1)

        @test floor(tau, tau[end]) == tau.size - 1
        @test floor(tau, β) == tau.size - 1
    end

    @testset "Log Grid for fermiK" begin
        kF, maxK, halfLife = 1.0, 3.0, 0.5
        K = Grid.fermiK(kF, maxK, halfLife, 16)
        # println(K.grid)
        @test floor(K, 0.0) == 1
        @test floor(K, K[1]) == 1

        δ = 1.0e-12
        check(K, 2:K.size - 1, δ, 0)
        check(K, 2:K.size - 1, -δ, -1)

        @test floor(K, K[end] - δ) == K.size - 1
        @test floor(K, K[end]) == K.size - 1
        @test floor(K, maxK) == K.size - 1
    end

    @testset "Log Grid for boseK" begin
        kF, maxK, halfLife = 1.0, 3.0, 0.5
        K = Grid.boseK(kF, maxK, halfLife, 16)
        # println(K.grid)
        @test floor(K, 0.0) == 1
        @test floor(K, K[1]) == 1

        δ = 1.0e-12
        check(K, 2:K.size - 1, δ, 0)
        check(K, 2:K.size - 1, -δ, -1)

        @test floor(K, K[end] - δ) == K.size - 1
        @test floor(K, K[end]) == K.size - 1
        @test floor(K, maxK) == K.size - 1
    end

end

@testset "UniLog Grids" begin
    function check(grid, range, shift, idx_shift)
        for i in range
            @test(floor(grid, grid[i] + shift) == i + idx_shift)
            # if floor(grid, grid[i] + shift) != i + idx_shift
            #     return false
            # end
        end
        return true
    end

    @testset "UniLog" begin
        EPS = 1e-9
        bound = @SVector[1.0, 3.0]
        init = 4
        minterval = 0.01
        M = 2
        N = 3
        isopen = @SVector[false,false]
        d2s = true
        g = Grid.UniLog{Float64}(bound, init, minterval, M, N, d2s,isopen)

        val = Grid._grid(g,init)
        @test Grid._floor(g,val-EPS) == init
        @test Grid._floor(g,val+EPS) == init
        for i = init+1:(M+1)*N+init
            val = Grid._grid(g,i)
            # @printf("%10.6f , %10.6f\n", i, val)
            # @printf("\t%10.6f , %10.6f\n", val-EPS, Grid._floor(g,val-EPS))
            # @printf("\t%10.6f , %10.6f\n", val+EPS, Grid._floor(g,val+EPS))
            @test Grid._floor(g,val-EPS) == i-1
            @test Grid._floor(g,val+EPS) == i
        end
    end

    @testset "UniLogs" begin
        EPS = 1e-9
        seg = 4
        bounds = @SVector[0.0,1.0,2.0,3.0]
        M=2
        N=3
        minterval = 0.01
        isopen = @SVector[false,false]
        g = Grid.UniLogs{Float64,(M+1)*N*seg+1,seg}(bounds,minterval,M,N,isopen,[true,true])

        val = g.grid[1]
        @test floor(g,val-EPS) == 1
        @test floor(g,val+EPS) == 1
        for i = 2:g.size-1
            val = g.grid[i]
            # @printf("%10.6f , %10.6f\n", i, val)
            # @printf("\t%10.6f , %10.6f\n", val-EPS, floor(g,val-EPS))
            # @printf("\t%10.6f , %10.6f\n", val+EPS, floor(g,val+EPS))
            @test floor(g,val-EPS) == i-1
            @test floor(g,val+EPS) == i
        end
        val = g.grid[g.size]
        @test floor(g,val-EPS) == g.size-1
        @test floor(g,val+EPS) == g.size-1

        for i =1:seg-1
            val = g.segment[i]
            # @printf("%10.6f , %10.6f\n", g.segindex[i], val)
            # @printf("\t%10.6f , %10.6f\n", val-EPS, floor(g,val-EPS))
            # @printf("\t%10.6f , %10.6f\n", val+EPS, floor(g,val+EPS))
            @test floor(g,val-EPS) == g.segindex[i]-1
            @test floor(g,val+EPS) == g.segindex[i]
        end
    end

    @testset "UniLog Grid for Tau" begin
        β = 10.0
        tau = Grid.tauUL(β, 0.001, 2,3)
        @test floor(tau, 0.0) == 1
        @test floor(tau, tau[1]) == 1 # tau[1]=0^+ is special

        δ = 1.0e-12
        check(tau, 2:tau.size - 1, δ, 0)
        check(tau, 2:tau.size - 1, -δ, -1)

        @test floor(tau, tau[end]) == tau.size - 1
        @test floor(tau, β) == tau.size - 1
    end

    @testset "UniLog Grid for fermiK" begin
        kF, maxK, minterval = 1.0, 3.0, 0.001
        K = Grid.fermiKUL(kF, maxK, minterval, 2,3)
        # println(K.grid)
        @test floor(K, 0.0) == 1
        @test floor(K, K[1]) == 1

        δ = 1.0e-12
        check(K, 2:K.size - 1, δ, 0)
        check(K, 2:K.size - 1, -δ, -1)

        @test floor(K, K[end] - δ) == K.size - 1
        @test floor(K, K[end]) == K.size - 1
        @test floor(K, maxK) == K.size - 1
    end

    @testset "UniLog Grid for boseK" begin
        kF, maxK, minterval = 1.0, 3.0, 0.001
        K = Grid.boseKUL(kF, maxK, minterval,2,3)
        # println(K.grid)
        @test floor(K, 0.0) == 1
        @test floor(K, K[1]) == 1

        δ = 1.0e-12
        check(K, 2:K.size - 1, δ, 0)
        check(K, 2:K.size - 1, -δ, -1)

        @test floor(K, K[end] - δ) == K.size - 1
        @test floor(K, K[end]) == K.size - 1
        @test floor(K, maxK) == K.size - 1
    end

end

@testset "Interpolate" begin
    @testset "Linear1D" begin
        β = π
        N,M= 3, 4
        tgrid = Grid.tauUL(β, 0.01β, M, N)
        # tugrid = Grid.Uniform{Float64,33}(0.0, β, (true, true))
        # kugrid = Grid.Uniform{Float64,33}(0.0, maxK, (true, true))
        f(t) = t
        data = zeros(tgrid.size)

        for (ti, t) in enumerate(tgrid.grid)
            data[ti] = f(t)
        end

        for ti = 1:tgrid.size - 1
            t = tgrid[ti] + 1.e-6
            fbar = Grid.linear1D(data, tgrid, t)
            @test abs(f(tgrid[ti]) - fbar) < 3.e-6 # linear interpolation, so error is δK+δt
            @test f(tgrid[ti]) < fbar
            @test f(tgrid[ti + 1]) > fbar
        end
        for ti = 2:tgrid.size
            t = tgrid[ti] - 1.e-6
            fbar = Grid.linear1D(data, tgrid, t)
            @test abs(f(tgrid[ti]) - fbar) < 3.e-6 # linear interpolation, so error is δK+δt
            @test f(tgrid[ti]) > fbar
            @test f(tgrid[ti - 1]) < fbar
        end

        t = tgrid[1] + eps(Float64)*1e3
        fbar = Grid.linear1D(data, tgrid, t)
        @test abs(f(t) - fbar) < 3.e-6 # linear interpolation, so error is δK+δt

        t = tgrid[tgrid.size] - eps(Float64)*1e3
        fbar = Grid.linear1D(data, tgrid, t)
        @test abs(f(t) - fbar) < 3.e-6 # linear interpolation, so error is δK+δt

        tlist = rand(10) * β
        # println(tlist)

        for (ti, t) in enumerate(tlist)
            fbar = Grid.linear1D(data, tgrid, t)
            # println("$k, $t, $fbar, ", f(k, t))
            @test abs(f(t) - fbar) < 3.e-6 # linear interpolation, so error is δK+δt
        end
    end

    @testset "Linear2D" begin
        β, kF, maxK = 10.0, 1.0, 3.0
        Nt, Nk = 8, 7
        tgrid = Grid.tau(β, 0.2β, Nt)
        kgrid = Grid.boseK(kF, maxK, 0.2kF, Nk)
        # tugrid = Grid.Uniform{Float64,33}(0.0, β, (true, true))
        # kugrid = Grid.Uniform{Float64,33}(0.0, maxK, (true, true))
        f(k, t) = k + t * 1.1
        data = zeros((Nk, Nt))

        for (ti, t) in enumerate(tgrid.grid)
            for (ki, k) in enumerate(kgrid.grid)
                data[ki, ti] = f(k, t)
            end
        end

        for ti = 1:tgrid.size - 1
            for ki = 1:kgrid.size - 1
                t = tgrid[ti] + 1.e-6
                k = kgrid[ki] + 1.e-6
                fbar = Grid.linear2D(data, kgrid, tgrid, k, t)
                @test abs(f(kgrid[ki], tgrid[ti]) - fbar) < 3.e-6 # linear interpolation, so error is δK+δt
                @test f(kgrid[ki], tgrid[ti]) < fbar
                @test f(kgrid[ki], tgrid[ti + 1]) > fbar
                @test f(kgrid[ki + 1], tgrid[ti]) > fbar
                @test f(kgrid[ki + 1], tgrid[ti + 1]) > fbar
            end
        end

        for ti = 2:tgrid.size
            for ki = 2:kgrid.size
                t = tgrid[ti] - 1.e-6
                k = kgrid[ki] - 1.e-6
                fbar = Grid.linear2D(data, kgrid, tgrid, k, t)
                @test abs(f(kgrid[ki], tgrid[ti]) - fbar) < 3.e-6 # linear interpolation, so error is δK+δt
                @test f(kgrid[ki], tgrid[ti]) > fbar
                @test f(kgrid[ki], tgrid[ti - 1]) < fbar
                @test f(kgrid[ki - 1], tgrid[ti]) < fbar
                @test f(kgrid[ki - 1], tgrid[ti - 1]) < fbar
            end
        end

        tlist = rand(10) * β
        klist = rand(10) * maxK
        # println(tlist)

        for (ti, t) in enumerate(tlist)
            for (ki, k) in enumerate(klist)
                fbar = Grid.linear2D(data, kgrid, tgrid, k, t)
                # println("$k, $t, $fbar, ", f(k, t))
                @test abs(f(k, t) - fbar) < 3.e-6 # linear interpolation, so error is δK+δt
            end
        end
    end

    @testset "TestInterpolation1D for tauUL" begin
        β, minterval = 1.0, 0.0001
        M, N = 11, 4
        tgrid1 = Grid.tauUL(β, minterval, M, N)
        tgrid2 = Grid.tauUL(β, minterval, 2M, 2N)
        # tugrid = Grid.Uniform{Float64,33}(0.0, β, (true, true))
        # kugrid = Grid.Uniform{Float64,33}(0.0, maxK, (true, true))
        f(t) = Spectral.kernelFermiT(t/β,0.01)

        d_max, std = Grid.testInterpolation1D(f, tgrid1, tgrid2, true)
        println("Testing interpolation for tauUL grid")
        println("d_max=",d_max,"\t std=", std)
    end

    @testset "TestInterpolation1D for fermiKUL" begin
        M, N= 15, 8
        kF, maxK, minterval = 1.0, 3.0, 0.00001
        kgrid1 = Grid.fermiKUL(kF, maxK, minterval, M, N)
        kgrid2 = Grid.fermiKUL(kF, maxK, minterval, 2M, 2N)
        # tugrid = Grid.Uniform{Float64,33}(0.0, β, (true, true))
        # kugrid = Grid.Uniform{Float64,33}(0.0, maxK, (true, true))
        f(k) = 1.0/(0.0004π^2+(k^2-kF^2)^2)

        d_max, std = Grid.testInterpolation1D(f, kgrid1, kgrid2, true)
        println("Testing interpolation for fermiKUL grid")
        println("d_max=",d_max,"\t std=", std)
    end

    @testset "Optimize UniLog grid" begin
        struct Para
            β::Float64
            τ_min::Float64
            k_min::Float64
            k_max::Float64
            kF::Float64

            function Para()
                return new(10.0, 0.001, 0.0001, 3.0, 1.0)
            end
        end

        para = Para()
        MN = 64

        println("Testing optimization of tauUL")
        f(t) = Spectral.kernelFermiT(t/para.β,1e0*para.kF^2*para.β)
        M, N, d_max = Grid.optimizeUniLog(Grid.tauUL, para, MN, f)
        println(MN,"\t", M,"\t", N,"\t", d_max)

        tgrid1 = Grid.tauUL(para, M, N)
        tgrid2 = Grid.tauUL(para, 2M, 2N)
        d_max, std = Grid.testInterpolation1D(f, tgrid1, tgrid2, true)
        println("Testing interpolation for tauUL grid")
        println("d_max=",d_max,"\t std=", std)


        MN = 64
        println("Testing optimization of fermiKUL")
        f2(k) = 1.0/((2π/para.β)^2+(k^2-para.kF^2)^2)
        M, N, d_max= Grid.optimizeUniLog(Grid.fermiKUL, para, MN, f2)
        println(MN,"\t", M,"\t", N,"\t", d_max)

        kgrid1 = Grid.fermiKUL(para, M, N)
        kgrid2 = Grid.fermiKUL(para, 2M, 2N)
        d_max, std, index = Grid.testInterpolation1D(f2, kgrid1, kgrid2, true, true)
        println("Testing interpolation for fermiKUL grid")
        println("d_max=",d_max,"\t std=", std)
        # println("index=", index, "\t x=", kgrid2[index])
        # index1=(floor(kgrid1, kgrid2[index]))
        # println(kgrid1[index1],"\t",kgrid1[index1+1])
        # println(kgrid1.grid)

    end

end

