@testset "CPU SimpleHT" begin
    @testset "Empty table" begin
        keys = UInt32[1]
        vals = UInt32[100]
        table = CPUSimpleHT(keys, vals)

        found, val = query(table, UInt32(999))
        @test found == false
    end

    @testset "Single entry" begin
        keys = UInt32[42]
        vals = UInt32[123]
        table = CPUSimpleHT(keys, vals)

        found, val = query(table, UInt32(42))
        @test found == true
        @test val == UInt32(123)

        found, val = query(table, UInt32(999))
        @test found == false
    end

    @testset "Bulk insert and query" begin
        Random.seed!(12345)
        n = 100_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)

        table = CPUSimpleHT(keys, vals)

        for i in 1:min(n, 1000)
            found, val = query(table, keys[i])
            @test found == true
            @test val == vals[i]
        end
    end

    @testset "Negative queries" begin
        Random.seed!(54321)
        n = 10_000

        inserted_keys = unique(rand(UInt32(1):UInt32(2^30), n * 2))[1:n]
        vals = rand(UInt32, n)
        table = CPUSimpleHT(inserted_keys, vals)

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
        table = CPUSimpleHT(keys, vals; load_factor=0.5)

        found_vec, result_vec = query(table, keys)

        @test all(found_vec)
        @test result_vec == vals
    end

    @testset "Load factors" begin
        Random.seed!(22222)

        for load_factor in [0.3, 0.5, 0.7]
            n = 10_000
            keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
            vals = rand(UInt32, n)

            table = CPUSimpleHT(keys, vals; load_factor=load_factor)

            found_vec, result_vec = query(table, keys)
            @test all(found_vec)
            @test result_vec == vals
        end
    end

    @testset "Sentinel key rejection" begin
        keys = UInt32[GPUHashTables.EMPTY_KEY_U32]
        vals = UInt32[100]

        @test_throws ErrorException CPUSimpleHT(keys, vals)
    end

    @testset "Duplicate keys (update)" begin
        keys = UInt32[1, 2, 1]
        vals = UInt32[100, 200, 300]

        table = CPUSimpleHT(keys, vals)

        found, val = query(table, UInt32(1))
        @test found == true
        @test val == UInt32(300)  # last value wins

        found, val = query(table, UInt32(2))
        @test found == true
        @test val == UInt32(200)
    end

    @testset "Delete (tombstone)" begin
        Random.seed!(33333)
        n = 1_000

        keys = unique(rand(UInt32(1):UInt32(2^31), n * 2))[1:n]
        vals = rand(UInt32, n)
        table = CPUSimpleHT(keys, vals)

        # Delete first 100 keys
        for i in 1:100
            result = Base.delete!(table, keys[i])
            @test result == true
        end

        # Deleted keys should not be found
        for i in 1:100
            found, _ = query(table, keys[i])
            @test found == false
        end

        # Remaining keys should still be found with correct values
        for i in 101:n
            found, val = query(table, keys[i])
            @test found == true
            @test val == vals[i]
        end
    end

    @testset "Delete non-existent key" begin
        keys = UInt32[1, 2, 3]
        vals = UInt32[10, 20, 30]
        table = CPUSimpleHT(keys, vals)

        result = Base.delete!(table, UInt32(999))
        @test result == false

        # Table should be intact
        found, val = query(table, UInt32(1))
        @test found == true
        @test val == UInt32(10)
    end

    @testset "Re-insert after delete" begin
        keys = UInt32[1, 2, 3]
        vals = UInt32[10, 20, 30]
        table = CPUSimpleHT(keys, vals)

        # Delete key 2
        @test Base.delete!(table, UInt32(2)) == true
        found, _ = query(table, UInt32(2))
        @test found == false

        # Re-insert key 2 with a new value by rebuilding
        # (CPUSimpleHT has no in-place insert; re-insert via a new table)
        keys2 = UInt32[1, 2, 3]
        vals2 = UInt32[10, 99, 30]
        table2 = CPUSimpleHT(keys2, vals2)

        found, val = query(table2, UInt32(2))
        @test found == true
        @test val == UInt32(99)
    end
end
