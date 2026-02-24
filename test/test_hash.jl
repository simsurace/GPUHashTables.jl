@testset "Hash Functions" begin
    @testset "murmur_hash_64a consistency" begin
        # Same input should always produce same output
        key = UInt32(12345)
        h1 = GPUHashTables.murmur_hash_64a(key)
        h2 = GPUHashTables.murmur_hash_64a(key)
        @test h1 == h2

        # Different seeds should produce different hashes
        h_seed0 = GPUHashTables.murmur_hash_64a(key, UInt64(0))
        h_seed1 = GPUHashTables.murmur_hash_64a(key, UInt64(1))
        @test h_seed0 != h_seed1
    end

    @testset "double_hash properties" begin
        # Both components should be non-zero for typical keys
        for _ in 1:100
            key = rand(UInt32(1):UInt32(2^31))
            h1, h2 = double_hash(key)
            # h1 and h2 should generally be different
            # (not a strict requirement, but good property)
            @test !(h1 == 0 && h2 == 0)
        end
    end

    @testset "double_hash distribution" begin
        # Check that hash values are reasonably distributed
        n_samples = 10000
        keys = rand(UInt32(1):UInt32(2^31), n_samples)

        h1_values = Set{UInt32}()
        h2_values = Set{UInt32}()

        for key in keys
            h1, h2 = double_hash(key)
            push!(h1_values, h1)
            push!(h2_values, h2)
        end

        # With good hash function, we should have many unique values
        # (at least 90% unique for random input)
        @test length(h1_values) > 0.9 * n_samples
        @test length(h2_values) > 0.9 * n_samples
    end

    @testset "double_hash determinism" begin
        # Verify hash is deterministic across calls
        keys = rand(UInt32(1):UInt32(2^31), 1000)
        hashes1 = [double_hash(k) for k in keys]
        hashes2 = [double_hash(k) for k in keys]
        @test hashes1 == hashes2
    end
end
