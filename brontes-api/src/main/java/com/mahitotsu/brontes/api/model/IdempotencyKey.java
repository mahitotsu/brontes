package com.mahitotsu.brontes.api.model;

import java.util.UUID;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

import lombok.Data;

@Table
@Data
public class IdempotencyKey {
    @Id
    private UUID idempotencyKey;
    private long payloadHash;
}
