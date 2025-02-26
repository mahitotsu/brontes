package com.mahitotsu.brontes.api.repository;

import java.util.Optional;
import java.util.UUID;
import java.util.stream.Stream;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import com.mahitotsu.brontes.api.entity.AccountTransaction;

public interface AccountTransactionRepository extends JpaRepository<AccountTransaction, UUID> {

    @Query("""
            SELECT atx FROM AccountTransaction atx
            WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber AND txSequence IS NULL
            ORDER BY txTimestamp
                        """)
    Stream<AccountTransaction> findAllUncommittedTransactions(Integer branchNumber, Integer accountNumber);

    @Query("""
            SELECT atx FROM AccountTransaction atx
            WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber AND txSequence IS NOT NULL
            ORDER BY txSequence
                        """)
    Stream<AccountTransaction> findAllCommittedTransactions(Integer branchNumber, Integer accountNumber);

    @Query("""
            SELECT atx FROM AccountTransaction atx
            WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber
            AND txSequence = (
                SELECT MAX(btx.txSequence) FROM AccountTransaction btx
                WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber
            )
                        """)
    Optional<AccountTransaction> findOneLastCommittedTransaction(Integer branchNumber, Integer accountNumber);
}
