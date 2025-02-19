package com.mahitotsu.brontes.api.repository;

import java.math.BigDecimal;
import java.util.UUID;

import org.springframework.data.r2dbc.repository.Query;
import org.springframework.data.r2dbc.repository.R2dbcRepository;

import com.mahitotsu.brontes.api.model.AccountTransaction;

import reactor.core.publisher.Mono;

public interface AccountTransactionRepository extends R2dbcRepository<AccountTransaction, UUID> {

    @Query("""
    SELECT new_balance 
    FROM account_transactions
    WHERE branch_number = :branchNumber AND account_number = :accountNumber
    ORDER BY tx_seq DESC
    LIMIT 1
            """)
    Mono<BigDecimal> queryBalance(final String branchNumber, final String accountNumber);

    @Query("""
    SELECT tx_seq
    FROM account_transactions
    WHERE branch_number = :branchNumber AND account_number = :accountNumber
    ORDER BY tx_seq DESC
    LIMIT 1
            """)
    Mono<BigDecimal> queryLastSeq(final String branchNumber, final String accountNumber);
}
