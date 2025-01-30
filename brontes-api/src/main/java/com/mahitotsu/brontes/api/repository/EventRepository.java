package com.mahitotsu.brontes.api.repository;

import java.util.UUID;

import org.springframework.data.repository.reactive.ReactiveCrudRepository;
import org.springframework.stereotype.Repository;

import com.mahitotsu.brontes.api.entity.EventEntity;

@Repository
public interface EventRepository extends ReactiveCrudRepository<EventEntity, UUID> {
    
}
