package com.mahitotsu.brontes.api.repository;

import java.util.UUID;

import org.springframework.data.r2dbc.repository.R2dbcRepository;

import com.mahitotsu.brontes.api.model.IdempotencyKey;

public interface IdempotencyKeyRepository extends R2dbcRepository<IdempotencyKey, UUID> {
    
}
