package com.mahitotsu.brontes.api.model;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

import lombok.Data;

@Table("account_transactions")
@Data
public class AccountTransaction {

    public static enum TxType {
        O, C, D, W
    }

    @Id
    private UUID txId;
    private Long txSeq;
    private Instant txTime;
    private TxType txType;
    private String branchNumber;
    private String accountNumber;
    private BigDecimal amount;
    private BigDecimal newBalance;
}
