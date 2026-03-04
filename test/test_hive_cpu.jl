@testset "CPU HiveHT" begin
    @testset "Empty table" begin
        keys = UInt32[1]
        vals = UInt32[100]
        table = CPUHiveHT(keys, vals)

        found, val = query(table, UInt32(999))
        @test found == false
    end

    @testset "Single entry" begin
        keys = UInt32[42]
        vals = UInt32[123]
        table = CPUHiveHT(keys, vals)

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

        table = CPUHiveHT(keys, vals)

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
        table = CPUHiveHT(inserted_keys, vals)

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
        table = CPUHiveHT(keys, vals; load_factor=0.5)

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

            table = CPUHiveHT(keys, vals; load_factor=load_factor)

            found_vec, result_vec = query(table, keys)
            @test all(found_vec)
            @test result_vec == vals
        end
    end

    @testset "Duplicate keys (update)" begin
        keys = UInt32[1, 2, 1]
        vals = UInt32[100, 200, 300]

        table = CPUHiveHT(keys, vals)

        found, val = query(table, UInt32(1))
        @test found == true
        @test val == UInt32(300)  # last value wins

        found, val = query(table, UInt32(2))
        @test found == true
        @test val == UInt32(200)
    end
end
