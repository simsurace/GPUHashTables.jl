@testset "CUDA SimpleHT" begin
    @testset "GPU matches CPU - small" begin
        Random.seed!(33333)
        n = 1000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUSimpleHT(keys, vals)
        gpu_table = CuSimpleHT(cpu_table)

        cpu_found, cpu_results = query(cpu_table, keys)
        gpu_found, gpu_results = query(gpu_table, keys)

        @test cpu_found == gpu_found
        @test cpu_results == gpu_results
    end

    @testset "GPU matches CPU - large" begin
        Random.seed!(44444)
        n = 100_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUSimpleHT(keys, vals)
        gpu_table = CuSimpleHT(cpu_table)

        cpu_found, cpu_results = query(cpu_table, keys)
        gpu_found, gpu_results = query(gpu_table, keys)

        @test cpu_found == gpu_found
        @test cpu_results == gpu_results
    end

    @testset "GPU negative queries" begin
        Random.seed!(55555)
        n = 10_000

        inserted_keys = unique(rand(UInt32(1):UInt32(2^30), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUSimpleHT(inserted_keys, vals)
        gpu_table = CuSimpleHT(cpu_table)

        query_keys = rand(UInt32(2^30 + 1):UInt32(2^31), 1000)

        cpu_found, _ = query(cpu_table, query_keys)
        gpu_found, _ = query(gpu_table, query_keys)

        @test cpu_found == gpu_found
        @test all(.!gpu_found)
    end

    @testset "GPU mixed queries" begin
        Random.seed!(66666)
        n = 10_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUSimpleHT(keys, vals)
        gpu_table = CuSimpleHT(cpu_table)

        existing_sample = keys[rand(1:n, 500)]
        nonexisting = rand(UInt32(1):UInt32(100), 500)
        mixed_keys = vcat(existing_sample, nonexisting)

        cpu_found, cpu_results = query(cpu_table, mixed_keys)
        gpu_found, gpu_results = query(gpu_table, mixed_keys)

        @test cpu_found == gpu_found

        for i in 1:length(mixed_keys)
            if cpu_found[i]
                @test cpu_results[i] == gpu_results[i]
            end
        end
    end

    @testset "GPU batch sizes" begin
        Random.seed!(77777)
        n = 50_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUSimpleHT(keys, vals)
        gpu_table = CuSimpleHT(cpu_table)

        for batch_size in [1, 7, 32, 100, 1000, 10_000]
            query_keys = keys[1:min(batch_size, n)]

            cpu_found, cpu_results = query(cpu_table, query_keys)
            gpu_found, gpu_results = query(gpu_table, query_keys)

            @test cpu_found == gpu_found
            @test cpu_results == gpu_results
        end
    end

    @testset "GPU with CuVector input" begin
        Random.seed!(88888)
        n = 10_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUSimpleHT(keys, vals)
        gpu_table = CuSimpleHT(cpu_table)

        gpu_keys = CuVector(keys)
        gpu_found, gpu_results = query(gpu_table, gpu_keys)

        cpu_found, cpu_results = query(cpu_table, keys)

        @test Vector(gpu_found) == cpu_found
        @test Vector(gpu_results) == cpu_results
    end

    @testset "GPU load factors" begin
        Random.seed!(99999)

        @testset for load_factor in [0.3, 0.5, 0.7]
            n = 10_000
            keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
            vals = rand(UInt32, n)

            cpu_table = CPUSimpleHT(keys, vals; load_factor=load_factor)
            gpu_table = CuSimpleHT(cpu_table)

            cpu_found, cpu_results = query(cpu_table, keys)
            gpu_found, gpu_results = query(gpu_table, keys)

            @test cpu_found == gpu_found
            @test cpu_results == gpu_results
        end
    end

    @testset "GPU reflects CPU tombstones" begin
        # Deleted entries on the CPU side must appear as not-found on the GPU,
        # while neighbouring keys in the same probe chain remain findable.
        Random.seed!(12121)
        n = 5_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUSimpleHT(keys, vals)

        # Delete every other key on the CPU side
        deleted = keys[1:2:end]
        kept    = keys[2:2:end]
        kept_vals = vals[2:2:end]

        for k in deleted
            Base.delete!(cpu_table, k)
        end

        # Upload the tombstoned table to GPU
        gpu_table = CuSimpleHT(cpu_table)

        # Deleted keys should not be found on the GPU
        gpu_found_del, _ = query(gpu_table, deleted)
        @test all(.!gpu_found_del)

        # Keys that were not deleted should still be found with correct values
        gpu_found_kept, gpu_results_kept = query(gpu_table, kept)
        @test all(gpu_found_kept)
        @test gpu_results_kept == kept_vals
    end
end
