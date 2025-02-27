package com.mahitotsu.brontes.api.repository;

import java.time.ZonedDateTime;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Stream;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import com.mahitotsu.brontes.api.entity.AccountTx;

public interface AccountTxRepository extends JpaRepository<AccountTx, UUID> {

    @Query("""
            SELECT atx FROM AccountTx atx
            WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber
                AND txSequence IS NULL AND txTimestamp <= :txTimestamp
            ORDER BY txTimestamp
            """)
    Stream<AccountTx> findUncommittedTransactions(Integer branchNumber, Integer accountNumber,
            ZonedDateTime txTimestamp);

    @Query("""
            SELECT atx FROM AccountTx atx
            WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber
            AND txSequence = (
                SELECT max(btx.txSequence) FROM AccountTx btx
                WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber
            )
            """)
    Optional<AccountTx> findLastCommittedTransaction(Integer branchNumber, Integer accountNumber);
}
