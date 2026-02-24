@testset "GPU Hash Table" begin
    @testset "GPU matches CPU - small" begin
        Random.seed!(33333)
        n = 1000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = DoubleHashTable(keys, vals)
        gpu_table = GPUDoubleHashTable(cpu_table)

        # Query same keys on both
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

        cpu_table = DoubleHashTable(keys, vals)
        gpu_table = GPUDoubleHashTable(cpu_table)

        # Query all keys
        cpu_found, cpu_results = query(cpu_table, keys)
        gpu_found, gpu_results = query(gpu_table, keys)

        @test cpu_found == gpu_found
        @test cpu_results == gpu_results
    end

    @testset "GPU negative queries" begin
        Random.seed!(55555)
        n = 10_000

        # Insert keys in low range
        inserted_keys = unique(rand(UInt32(1):UInt32(2^30), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = DoubleHashTable(inserted_keys, vals)
        gpu_table = GPUDoubleHashTable(cpu_table)

        # Query keys in high range (not inserted)
        query_keys = rand(UInt32(2^30 + 1):UInt32(2^31), 1000)

        cpu_found, _ = query(cpu_table, query_keys)
        gpu_found, _ = query(gpu_table, query_keys)

        @test cpu_found == gpu_found
        @test all(.!gpu_found)  # All should be not found
    end

    @testset "GPU mixed queries" begin
        Random.seed!(66666)
        n = 10_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = DoubleHashTable(keys, vals)
        gpu_table = GPUDoubleHashTable(cpu_table)

        # Mix of existing and non-existing keys
        existing_sample = keys[rand(1:n, 500)]
        nonexisting = rand(UInt32(1):UInt32(100), 500)  # Likely not in table
        mixed_keys = vcat(existing_sample, nonexisting)

        cpu_found, cpu_results = query(cpu_table, mixed_keys)
        gpu_found, gpu_results = query(gpu_table, mixed_keys)

        @test cpu_found == gpu_found

        # For found keys, values should match
        for i in 1:length(mixed_keys)
            if cpu_found[i]
                @test cpu_results[i] == gpu_results[i]
            end
        end
    end

    @testset "GPU batch sizes" begin
        Random.seed!(77777)

        # Build a table
        n = 50_000
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = DoubleHashTable(keys, vals)
        gpu_table = GPUDoubleHashTable(cpu_table)

        # Test various batch sizes
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

        cpu_table = DoubleHashTable(keys, vals)
        gpu_table = GPUDoubleHashTable(cpu_table)

        # Use GPU vectors directly
        gpu_keys = CuVector(keys)
        gpu_found, gpu_results = query(gpu_table, gpu_keys)

        # Compare with CPU
        cpu_found, cpu_results = query(cpu_table, keys)

        @test Vector(gpu_found) == cpu_found
        @test Vector(gpu_results) == cpu_results
    end

    @testset "GPU load factors" begin
        Random.seed!(99999)

        for load_factor in [0.5, 0.7, 0.9]
            n = 10_000
            keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
            vals = rand(UInt32, n)

            cpu_table = DoubleHashTable(keys, vals; load_factor=load_factor)
            gpu_table = GPUDoubleHashTable(cpu_table)

            cpu_found, cpu_results = query(cpu_table, keys)
            gpu_found, gpu_results = query(gpu_table, keys)

            @test cpu_found == gpu_found
            @test cpu_results == gpu_results
        end
    end
end
