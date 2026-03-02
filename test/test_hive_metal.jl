@testset "Metal HiveHT" begin
    @testset "Metal matches CPU - small" begin
        Random.seed!(33333)
        n = 1000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUHiveHT(keys, vals)
        metal_table = MtlHiveHT(cpu_table)

        # Query same keys on both
        cpu_found, cpu_results = query(cpu_table, keys)
        metal_found, metal_results = query(metal_table, keys)

        @test cpu_found == metal_found
        @test cpu_results == metal_results
    end

    @testset "Metal matches CPU - large" begin
        Random.seed!(44444)
        n = 100_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUHiveHT(keys, vals)
        metal_table = MtlHiveHT(cpu_table)

        # Query all keys
        cpu_found, cpu_results = query(cpu_table, keys)
        metal_found, metal_results = query(metal_table, keys)

        @test cpu_found == metal_found
        @test cpu_results == metal_results
    end

    @testset "Metal negative queries" begin
        Random.seed!(55555)
        n = 10_000

        # Insert keys in low range
        inserted_keys = unique(rand(UInt32(1):UInt32(2^30), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUHiveHT(inserted_keys, vals)
        metal_table = MtlHiveHT(cpu_table)

        # Query keys in high range (not inserted)
        query_keys = rand(UInt32(2^30 + 1):UInt32(2^31), 1000)

        cpu_found, _ = query(cpu_table, query_keys)
        metal_found, _ = query(metal_table, query_keys)

        @test cpu_found == metal_found
        @test all(.!metal_found)  # All should be not found
    end

    @testset "Metal mixed queries" begin
        Random.seed!(66666)
        n = 10_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUHiveHT(keys, vals)
        metal_table = MtlHiveHT(cpu_table)

        # Mix of existing and non-existing keys
        existing_sample = keys[rand(1:n, 500)]
        nonexisting = rand(UInt32(1):UInt32(100), 500)  # Likely not in table
        mixed_keys = vcat(existing_sample, nonexisting)

        cpu_found, cpu_results = query(cpu_table, mixed_keys)
        metal_found, metal_results = query(metal_table, mixed_keys)

        @test cpu_found == metal_found

        # For found keys, values should match
        for i in 1:length(mixed_keys)
            if cpu_found[i]
                @test cpu_results[i] == metal_results[i]
            end
        end
    end

    @testset "Metal batch sizes" begin
        Random.seed!(77777)

        # Build a table
        n = 50_000
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUHiveHT(keys, vals)
        metal_table = MtlHiveHT(cpu_table)

        # Test various batch sizes
        for batch_size in [1, 7, 32, 100, 1000, 10_000]
            query_keys = keys[1:min(batch_size, n)]

            cpu_found, cpu_results = query(cpu_table, query_keys)
            metal_found, metal_results = query(metal_table, query_keys)

            @test cpu_found == metal_found
            @test cpu_results == metal_results
        end
    end

    @testset "Metal with MtlVector input" begin
        Random.seed!(88888)
        n = 10_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUHiveHT(keys, vals)
        metal_table = MtlHiveHT(cpu_table)

        # Use Metal vectors directly
        metal_keys = MtlVector(keys)
        metal_found, metal_results = query(metal_table, metal_keys)

        # Compare with CPU
        cpu_found, cpu_results = query(cpu_table, keys)

        @test Vector(metal_found) == cpu_found
        @test Vector(metal_results) == cpu_results
    end

    @testset "Metal load factors" begin
        @testset for load_factor in [0.5, 0.7, 0.9]
            Random.seed!(99999)  # Reset seed for each load factor for reproducibility
            n = 10_000
            keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
            vals = rand(UInt32, n)

            cpu_table = CPUHiveHT(keys, vals; load_factor=load_factor)
            metal_table = MtlHiveHT(cpu_table)

            cpu_found, cpu_results = query(cpu_table, keys)
            metal_found, metal_results = query(metal_table, keys)

            @test cpu_found == metal_found
            @test cpu_results == metal_results
        end
    end

    @testset "Query" begin
        keys = rand(UInt32, 1000)
        vals = rand(UInt32, 1000)
        table = MtlHiveHT(keys, vals; load_factor = 0.1)
        found, results = query(table, mtl(keys))
        @test all(found)
    end
    
    @testset "Basic insert and query" begin
        Random.seed!(11111)

        # Create empty table
        n_buckets = 100  # 32 slots each = 3200 total slots
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert some keys
        n = 500
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

    @testset "Update existing keys" begin
        Random.seed!(22222)

        n_buckets = 100
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert initial data
        n = 500
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        status = upsert!(table, keys, vals)
        @test all(status .== UPSERT_INSERTED)

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

    @testset "Mixed insert and update" begin
        Random.seed!(33333)

        n_buckets = 200
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert initial data
        n_initial = 300
        initial_keys = unique(rand(UInt32(1):UInt32(2^30), n_initial * 2))[1:n_initial]
        initial_vals = rand(UInt32, n_initial)

        status = upsert!(table, initial_keys, initial_vals)
        @test all(status .== UPSERT_INSERTED)

        # Upsert mix of existing and new keys
        n_new = 200
        new_keys = unique(rand(UInt32(2^30 + 1):UInt32(2^31), n_new * 2))[1:n_new]
        new_vals = rand(UInt32, n_new)

        # Combine: update first half of initial, insert new
        n_update = 150
        upsert_keys = vcat(initial_keys[1:n_update], new_keys)
        upsert_vals = vcat(rand(UInt32, n_update), new_vals)

        status = upsert!(table, upsert_keys, upsert_vals)

        # First n_update should be updated, rest should be inserted
        @test all(status[1:n_update] .== UPSERT_UPDATED)
        @test all(status[n_update+1:end] .== UPSERT_INSERTED)

        # Verify updates
        found, results = query(table, upsert_keys)
        @test all(found)
        @test results == upsert_vals

        # Verify unchanged keys still have original values
        unchanged_keys = initial_keys[n_update+1:end]
        unchanged_vals = initial_vals[n_update+1:end]
        found2, results2 = query(table, unchanged_keys)
        @test all(found2)
        @test results2 == unchanged_vals
    end

    @testset "Negative queries" begin
        Random.seed!(44444)

        n_buckets = 100
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert keys in low range
        n = 500
        inserted_keys = unique(rand(UInt32(1):UInt32(2^30), n * 2))[1:n]
        vals = rand(UInt32, n)

        upsert!(table, inserted_keys, vals)

        # Query keys in high range (not inserted)
        query_keys = rand(UInt32(2^30 + 1):UInt32(2^31), 200)

        found, _ = query(table, query_keys)
        @test all(.!found)  # All should be not found
    end

    @testset "Delete existing keys" begin
        Random.seed!(55555)

        n_buckets = 100
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert data
        n = 500
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        upsert!(table, keys, vals)

        # Delete first 100 keys
        n_delete = 100
        delete_keys = keys[1:n_delete]
        status = delete!(table, delete_keys)

        @test all(status .== DELETE_SUCCESS)

        # Verify deleted keys are not found
        found, _ = query(table, delete_keys)
        @test all(.!found)

        # Verify remaining keys still exist
        remaining_keys = keys[n_delete+1:end]
        remaining_vals = vals[n_delete+1:end]
        found2, results2 = query(table, remaining_keys)
        @test all(found2)
        @test results2 == remaining_vals
    end

    @testset "Delete non-existing keys" begin
        Random.seed!(66666)

        n_buckets = 100
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert data in low range
        n = 300
        keys = unique(rand(UInt32(1):UInt32(2^30), n * 2))[1:n]
        vals = rand(UInt32, n)

        upsert!(table, keys, vals)

        # Try to delete keys in high range (not in table)
        delete_keys = rand(UInt32(2^30 + 1):UInt32(2^31), 100)
        status = delete!(table, delete_keys)

        @test all(status .== DELETE_FAILED)

        # Original data should be unaffected
        found, results = query(table, keys)
        @test all(found)
        @test results == vals
    end

    @testset "Slot reuse after delete" begin
        Random.seed!(77777)

        n_buckets = 50  # Small table to force slot reuse
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert data
        n = 500
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        status = upsert!(table, keys, vals)
        @test all(status .== UPSERT_INSERTED)

        # Delete half of the keys
        n_delete = 250
        delete_keys = keys[1:n_delete]
        status = delete!(table, delete_keys)
        @test all(status .== DELETE_SUCCESS)

        # Insert new keys - should reuse tombstone slots
        new_keys = unique(rand(UInt32(2^31):UInt32(typemax(UInt32) - 1), n_delete * 2))[1:n_delete]
        new_vals = rand(UInt32, n_delete)

        status = upsert!(table, new_keys, new_vals)
        @test all(status .== UPSERT_INSERTED)

        # Query new keys
        found, results = query(table, new_keys)
        @test all(found)
        @test results == new_vals

        # Query remaining old keys
        remaining_keys = keys[n_delete+1:end]
        remaining_vals = vals[n_delete+1:end]
        found2, results2 = query(table, remaining_keys)
        @test all(found2)
        @test results2 == remaining_vals
    end

    @testset "Batch sizes" begin
        Random.seed!(88888)

        n_buckets = 500
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert a bunch of keys
        n = 5000
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        upsert!(table, keys, vals)

        # Test various query batch sizes
        for batch_size in [1, 7, 32, 100, 1000, 5000]
            query_keys = keys[1:min(batch_size, n)]
            expected_vals = vals[1:min(batch_size, n)]

            found, results = query(table, query_keys)

            @test all(found)
            @test results == expected_vals
        end
    end

    @testset "Load factors" begin
        Random.seed!(99999)

        for load_factor in [0.5, 0.7, 0.9]
            n = 5000
            keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
            vals = rand(UInt32, n)

            table = MtlHiveHT(keys, vals; load_factor=load_factor)

            found, results = query(table, keys)
            @test all(found)
            @test results == vals
        end
    end

    @testset "MtlVector input" begin
        Random.seed!(10101)

        n_buckets = 100
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        n = 500
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        # Use GPU vectors directly
        gpu_keys = MtlVector(keys)
        gpu_vals = MtlVector(vals)

        status = upsert!(table, gpu_keys, gpu_vals)
        @test all(Vector(status) .== UPSERT_INSERTED)

        # Query with GPU vectors
        found, results = query(table, gpu_keys)
        @test all(Vector(found))
        @test Vector(results) == vals
    end

    @testset "Concurrent upserts - no data loss" begin
        Random.seed!(12121)

        # Create larger table for concurrent operations
        n_buckets = 1000
        table = MtlHiveHT{UInt32,UInt32}(n_buckets)

        # Insert many keys in batches
        total_keys = Vector{UInt32}()
        total_vals = Vector{UInt32}()

        for batch in 1:5
            n = 1000
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
