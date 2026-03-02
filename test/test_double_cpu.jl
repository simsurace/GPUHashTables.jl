@testset "CPU DoubleHT" begin
    @testset "Empty table" begin
        # Create table with single entry, then query for non-existent key
        keys = UInt32[1]
        vals = UInt32[100]
        table = CPUDoubleHT(keys, vals)

        found, val = query(table, UInt32(999))
        @test found == false
    end

    @testset "Single entry" begin
        keys = UInt32[42]
        vals = UInt32[123]
        table = CPUDoubleHT(keys, vals)

        # Query existing key
        found, val = query(table, UInt32(42))
        @test found == true
        @test val == UInt32(123)

        # Query non-existing key
        found, val = query(table, UInt32(999))
        @test found == false
    end

    @testset "Bulk insert and query" begin
        Random.seed!(12345)
        n = 100_000

        # Generate unique keys (avoid sentinel value)
        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        table = CPUDoubleHT(keys, vals)

        # All inserted keys should be found with correct values
        for i in 1:min(n, 1000)  # Test subset for speed
            found, val = query(table, keys[i])
            @test found == true
            @test val == vals[i]
        end
    end

    @testset "Negative queries" begin
        Random.seed!(54321)
        n = 10_000

        # Insert some keys
        inserted_keys = unique(rand(UInt32(1):UInt32(2^30), n * 2))[1:n]
        vals = rand(UInt32, n)
        table = CPUDoubleHT(inserted_keys, vals)

        # Generate keys that weren't inserted (high range)
        query_keys = rand(UInt32(2^30 + 1):UInt32(2^31), 1000)

        for key in query_keys
            if !(key in inserted_keys)
                found, _ = query(table, key)
                @test found == false
            end
        end
    end

    @testset "Batch query" begin
        Random.seed!(11111)
        n = 10_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)
        table = CPUDoubleHT(keys, vals; load_factor=0.5)

        # Batch query all keys
        found_vec, result_vec = query(table, keys)

        @test all(found_vec)
        @test result_vec == vals
    end

    @testset "Load factors" begin
        Random.seed!(22222)

        for load_factor in [0.5, 0.7, 0.9]
            n = 10_000
            keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
            vals = rand(UInt32, n)

            table = CPUDoubleHT(keys, vals; load_factor=load_factor)

            # Verify all keys are findable
            found_vec, result_vec = query(table, keys)
            @test all(found_vec)
            @test result_vec == vals
        end
    end

    @testset "Sentinel key rejection" begin
        # Should error if trying to insert sentinel key
        keys = UInt32[GPUHashTables.EMPTY_KEY_U32]
        vals = UInt32[100]

        @test_throws ErrorException CPUDoubleHT(keys, vals)
    end

    @testset "Duplicate keys (update)" begin
        # When same key is inserted twice, last value wins
        keys = UInt32[1, 2, 1]  # Key 1 appears twice
        vals = UInt32[100, 200, 300]

        table = CPUDoubleHT(keys, vals)

        found, val = query(table, UInt32(1))
        @test found == true
        @test val == UInt32(300)  # Last value for key 1

        found, val = query(table, UInt32(2))
        @test found == true
        @test val == UInt32(200)
    end
end
