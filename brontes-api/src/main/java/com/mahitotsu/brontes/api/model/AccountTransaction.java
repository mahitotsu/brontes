package com.mahitotsu.brontes.api.model;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

import lombok.Data;

@Table
@Data
public class AccountTransaction {
    @Id
    private UUID txId;
    private Long txSeq;
    private Instant txTime;
    private Character txType;
    private String branchNUmber;
    private String accountNumber;
    private BigDecimal amount;
    private BigDecimal newBigDecimal;
}
