@testset "Metal DoubleHT" begin
    @testset "Metal matches CPU - small" begin
        Random.seed!(33333)
        n = 1000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        cpu_table = CPUDoubleHT(keys, vals)
        metal_table = MtlDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(keys, vals)
        metal_table = MtlDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(inserted_keys, vals)
        metal_table = MtlDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(keys, vals)
        metal_table = MtlDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(keys, vals)
        metal_table = MtlDoubleHT(cpu_table)

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

        cpu_table = CPUDoubleHT(keys, vals)
        metal_table = MtlDoubleHT(cpu_table)

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

            cpu_table = CPUDoubleHT(keys, vals; load_factor=load_factor)
            metal_table = MtlDoubleHT(cpu_table)

            cpu_found, cpu_results = query(cpu_table, keys)
            metal_found, metal_results = query(metal_table, keys)

            @test cpu_found == metal_found
            @test cpu_results == metal_results
        end
    end
end

# @testset "Metal mutable DoubleHT" begin
#     @testset "Basic upsert - insert into empty table" begin
#         Random.seed!(11111)

#         # Create empty mutable table with plenty of buckets
#         n_buckets = 200  # 32 slots each = 6400 total slots
#         table = MtlDoubleHT{UInt32,UInt32}(n_buckets)

#         n = 1000
#         keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
#         vals = rand(UInt32, n)

#         # Insert in sub-batches due to Metal's relaxed memory model
#         # Each batch completes and synchronizes before the next
#         batch_size = 128
#         for i in 1:batch_size:n
#             batch_end = min(i + batch_size - 1, n)
#             status = upsert!(table, keys[i:batch_end], vals[i:batch_end])
#             @test all(status .== UPSERT_INSERTED)
#         end

#         # Query back
#         found, results = query(table, keys)
#         @test all(found)
#         @test results == vals
#     end

#     @testset "Upsert - update existing keys" begin
#         Random.seed!(22222)

#         # Create table with initial data
#         n = 500
#         keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
#         vals = rand(UInt32, n)

#         cpu_table = CPUDoubleHT(keys, vals)
#         table = MtlDoubleHT(cpu_table)

#         # Update with new values in sub-batches
#         new_vals = rand(UInt32, n)
#         batch_size = 128
#         for i in 1:batch_size:n
#             batch_end = min(i + batch_size - 1, n)
#             status = upsert!(table, keys[i:batch_end], new_vals[i:batch_end])
#             @test all(status .== UPSERT_UPDATED)
#         end

#         # Query back - should get new values
#         found, results = query(table, keys)
#         @test all(found)
#         @test results == new_vals
#     end

#     @testset "Upsert - mixed insert and update" begin
#         Random.seed!(33333)

#         # Create table with some initial data
#         n_initial = 250
#         n_new = 250
#         n_buckets = 100  # 32 slots each = 3200 total slots

#         initial_keys = unique(rand(UInt32(1):UInt32(2^30), n_initial * 2))[1:n_initial]
#         initial_vals = rand(UInt32, n_initial)

#         # Create empty mutable table with enough capacity
#         table = MtlDoubleHT{UInt32,UInt32}(n_buckets)

#         # Insert initial data in sub-batches
#         batch_size = 64
#         for i in 1:batch_size:n_initial
#             batch_end = min(i + batch_size - 1, n_initial)
#             upsert!(table, initial_keys[i:batch_end], initial_vals[i:batch_end])
#         end

#         # Upsert mix of existing and new keys
#         new_keys = unique(rand(UInt32(2^30 + 1):UInt32(2^31), n_new * 2))[1:n_new]
#         new_vals = rand(UInt32, n_new)

#         # Combine: update first half of initial, insert new
#         n_update = 125
#         upsert_keys = vcat(initial_keys[1:n_update], new_keys)
#         upsert_vals = vcat(rand(UInt32, n_update), new_vals)

#         # Upsert in sub-batches
#         all_status = UInt8[]
#         for i in 1:batch_size:length(upsert_keys)
#             batch_end = min(i + batch_size - 1, length(upsert_keys))
#             status = upsert!(table, upsert_keys[i:batch_end], upsert_vals[i:batch_end])
#             append!(all_status, status)
#         end

#         # First n_update should be updated, rest should be inserted
#         @test all(all_status[1:n_update] .== UPSERT_UPDATED)
#         @test all(all_status[n_update+1:end] .== UPSERT_INSERTED)

#         # Verify updates
#         found, results = query(table, upsert_keys)
#         @test all(found)
#         @test results == upsert_vals

#         # Verify unchanged keys still have original values
#         unchanged_keys = initial_keys[n_update+1:end]
#         unchanged_vals = initial_vals[n_update+1:end]
#         found2, results2 = query(table, unchanged_keys)
#         @test all(found2)
#         @test results2 == unchanged_vals
#     end

#     @testset "Upsert with MtlVector input" begin
#         Random.seed!(44444)

#         n_buckets = 100  # 32 slots each = 3200 total slots
#         table = MtlDoubleHT{UInt32,UInt32}(n_buckets)

#         # Smaller batch for single-call upsert
#         n = 64
#         keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
#         vals = rand(UInt32, n)

#         # Use GPU vectors directly
#         metal_keys = MtlVector(keys)
#         metal_vals = MtlVector(vals)
#         status = upsert!(table, metal_keys, metal_vals)

#         @test all(Vector(status) .== UPSERT_INSERTED)

#         # Query back
#         found, results = query(table, metal_keys)
#         @test all(Vector(found))
#         @test Vector(results) == vals
#     end

#     @testset "Concurrent upserts - no data loss" begin
#         Random.seed!(55555)

#         # Create larger table for concurrent operations
#         n_buckets = 5000
#         table = MtlDoubleHT{UInt32,UInt32}(n_buckets)

#         # Insert many keys in batches
#         # Use smaller sub-batches (256) to avoid Metal memory ordering issues
#         total_keys = Vector{UInt32}()
#         total_vals = Vector{UInt32}()

#         for batch in 1:5
#             n = 2000
#             keys = unique(rand(UInt32(batch * 10^7):UInt32((batch + 1) * 10^7 - 1), n * 2))[1:n]
#             vals = fill(UInt32(batch), n)  # Mark with batch number

#             # Insert in smaller sub-batches for reliability
#             sub_batch_size = 256
#             all_inserted = true
#             for i in 1:sub_batch_size:n
#                 sub_end = min(i + sub_batch_size - 1, n)
#                 status = upsert!(table, keys[i:sub_end], vals[i:sub_end])
#                 if !all(status .== UPSERT_INSERTED)
#                     all_inserted = false
#                 end
#             end
#             @test all_inserted

#             append!(total_keys, keys)
#             append!(total_vals, vals)
#         end

#         # Query all keys - all should be found with correct values
#         found, results = query(table, total_keys)
#         @test all(found)
#         @test results == total_vals
#     end
# end
