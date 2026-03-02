@testset "GPU Hash Table" begin
    @testset "GPU matches CPU - small" begin
        Random.seed!(33333)
        n = 1000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUDoubleHT(keys, vals)
        gpu_table = CuDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(keys, vals)
        gpu_table = CuDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(inserted_keys, vals)
        gpu_table = CuDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(keys, vals)
        gpu_table = CuDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(keys, vals)
        gpu_table = CuDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(keys, vals)
        gpu_table = CuDoubleHT(cpu_table)

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

        @testset for load_factor in [0.5, 0.7, 0.9]
            n = 10_000
            keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
            vals = rand(UInt32, n)

            cpu_table = CPUDoubleHT(keys, vals; load_factor=load_factor)
            gpu_table = CuDoubleHT(cpu_table)

            cpu_found, cpu_results = query(cpu_table, keys)
            gpu_found, gpu_results = query(gpu_table, keys)

            @test cpu_found == gpu_found
            @test cpu_results == gpu_results
        end
    end
end

@testset "Mutable GPU Hash Table (Upserts)" begin
    @testset "Basic upsert - insert into empty table" begin
        Random.seed!(11111)

        # Create empty mutable table
        n_buckets = 1000
        table = CuMutableDoubleHT{UInt32,UInt32}(n_buckets)

        # Insert some keys
        n = 1000
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        status = upsert!(table, keys, vals)

        # All should be inserted
        @test all(status .== UPSERT_INSERTED)

        # Query back
        found, results = query(table, keys)
        @test all(found)
        @test results == vals
    end

    @testset "Upsert - update existing keys" begin
        Random.seed!(22222)

        # Create table with initial data
        n = 1000
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUDoubleHT(keys, vals)
        table = CuMutableDoubleHT(cpu_table)

        # Update with new values
        new_vals = rand(UInt32, n)
        status = upsert!(table, keys, new_vals)

        # All should be updated
        @test all(status .== UPSERT_UPDATED)

        # Query back - should get new values
        found, results = query(table, keys)
        @test all(found)
        @test results == new_vals
    end

    @testset "Upsert - mixed insert and update" begin
        Random.seed!(33333)

        # Create table with some initial data
        n_initial = 500
        initial_keys = unique(rand(UInt32(1):UInt32(2^30), n_initial * 2))[1:n_initial]
        initial_vals = rand(UInt32, n_initial)

        cpu_table = CPUDoubleHT(initial_keys, initial_vals)
        table = CuMutableDoubleHT(cpu_table)

        # Upsert mix of existing and new keys
        n_new = 500
        new_keys = unique(rand(UInt32(2^30 + 1):UInt32(2^31), n_new * 2))[1:n_new]
        new_vals = rand(UInt32, n_new)

        # Combine: update first half of initial, insert new
        upsert_keys = vcat(initial_keys[1:250], new_keys)
        upsert_vals = vcat(rand(UInt32, 250), new_vals)

        status = upsert!(table, upsert_keys, upsert_vals)

        # First 250 should be updated, rest should be inserted
        @test all(status[1:250] .== UPSERT_UPDATED)
        @test all(status[251:end] .== UPSERT_INSERTED)

        # Verify updates
        found, results = query(table, upsert_keys)
        @test all(found)
        @test results == upsert_vals

        # Verify unchanged keys still have original values
        unchanged_keys = initial_keys[251:end]
        unchanged_vals = initial_vals[251:end]
        found2, results2 = query(table, unchanged_keys)
        @test all(found2)
        @test results2 == unchanged_vals
    end

    @testset "Upsert with CuVector input" begin
        Random.seed!(44444)

        n_buckets = 500
        table = CuMutableDoubleHT{UInt32,UInt32}(n_buckets)

        n = 500
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        # Use GPU vectors directly
        gpu_keys = CuVector(keys)
        gpu_vals = CuVector(vals)
        status = upsert!(table, gpu_keys, gpu_vals)

        @test all(Vector(status) .== UPSERT_INSERTED)

        # Query back
        found, results = query(table, gpu_keys)
        @test all(Vector(found))
        @test Vector(results) == vals
    end

    @testset "Concurrent upserts - no data loss" begin
        Random.seed!(55555)

        # Create larger table for concurrent operations
        n_buckets = 5000
        table = CuMutableDoubleHT{UInt32,UInt32}(n_buckets)

        # Insert many keys in batches
        total_keys = Vector{UInt32}()
        total_vals = Vector{UInt32}()

        for batch in 1:5
            n = 2000
            keys = unique(rand(UInt32(batch * 10^7):UInt32((batch + 1) * 10^7 - 1), n * 2))[1:n]
            vals = fill(UInt32(batch), n)  # Mark with batch number

            status = upsert!(table, keys, vals)
            @test all(status .== UPSERT_INSERTED)

            append!(total_keys, keys)
            append!(total_vals, vals)
        end

        # Query all keys - all should be found with correct values
        found, results = query(table, total_keys)
        @test all(found)
        @test results == total_vals
    end
end
