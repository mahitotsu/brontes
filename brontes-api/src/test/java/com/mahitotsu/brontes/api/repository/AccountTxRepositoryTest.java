package com.mahitotsu.brontes.api.repository;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.math.BigDecimal;
import java.time.ZonedDateTime;
import java.util.Random;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.transaction.support.TransactionOperations;

import com.mahitotsu.brontes.api.AbstractTestBase;
import com.mahitotsu.brontes.api.entity.AccountTx;
import com.mahitotsu.brontes.api.entity.AccountTx.TxStatus;

public class AccountTxRepositoryTest extends AbstractTestBase {

    private static final Random RANDOM = new Random();

    @Autowired
    private AccountTxRepository accountTxRepository;

    @Autowired
    private TransactionOperations txOperations;

    private Integer randomBranchNumber() {
        return RANDOM.nextInt(1000);
    }

    private Integer randomAccountNumber() {
        return RANDOM.nextInt(10000000);
    }

    private BigDecimal randomAmount() {
        return new BigDecimal(RANDOM.nextDouble() * 10000000000000L);
    }

    @Test
    public void testInsertNewEntity() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        final AccountTx newEntity = AccountTx.newEntity(branchNumber, accuntNumber, amount);
        assertNull(newEntity.getTxId());

        final AccountTx savedEntity = this.accountTxRepository.save(newEntity);
        assertNotNull(savedEntity.getTxId());
        assertNull(savedEntity.getTxTimestamp());

        final AccountTx foundEntity = this.accountTxRepository.findById(savedEntity.getTxId()).get();
        assertNotNull(foundEntity.getTxTimestamp());
        assertEquals(AccountTx.TxStatus.REGISTERED, foundEntity.getTxStatus());
        assertNull(foundEntity.getTxSequence());
        assertNull(foundEntity.getNewBalance());
    }

    @Test
    public void testInsertNewEntity_NullBranchNumber() {

        final Integer branchNumber = null;
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.accountTxRepository
                .save(AccountTx.newEntity(branchNumber, accuntNumber, amount)));
    }

    @Test
    public void testInsertNewEntity_InvalidLengthBranchNumber() {

        final Integer branchNumber = this.randomBranchNumber() + 1000;
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.accountTxRepository
                .save(AccountTx.newEntity(branchNumber, accuntNumber, amount)));
    }

    @Test
    public void testInsertNewEntity_NullAccountNumber() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = null;
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.accountTxRepository
                .save(AccountTx.newEntity(branchNumber, accuntNumber, amount)));
    }

    @Test
    public void testInsertNewEntity_InvalidLengthAccountNumber() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber() + 10000000;
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.accountTxRepository
                .save(AccountTx.newEntity(branchNumber, accuntNumber, amount)));
    }

    @Test
    public void testLastCommittedAccountTx_NoTx() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();

        final AccountTx lastCommittedTx = this.accountTxRepository
                .findLastCommittedTransaction(branchNumber, accuntNumber).orElse(null);
        assertNull(lastCommittedTx);
    }

    @Test
    public void testFirstUncommittedAccountTx_NoTx() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();

        final AccountTx firstUncommittedTx = this.txOperations.execute(tx -> this.accountTxRepository
                .findUncommittedTransactions(branchNumber, accuntNumber, ZonedDateTime.now()).findFirst().orElse(null));
        assertNull(firstUncommittedTx);
    }

    @Test
    public void testInsertNewEntity_Acceptable() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        // register initial entity
        final UUID txId = this.accountTxRepository
                .save(AccountTx.newEntity(branchNumber, accuntNumber, amount)).getTxId();
        final AccountTx initialEntity = this.accountTxRepository.findById(txId).get();

        // before commit the transaction
        final AccountTx firstUncommittedTx = this.txOperations.execute(tx -> this.accountTxRepository
                .findUncommittedTransactions(branchNumber, accuntNumber, ZonedDateTime.now()).findFirst().orElse(null));
        assertNotNull(firstUncommittedTx);
        assertEquals(initialEntity, firstUncommittedTx);

        final AccountTx lastCommittedTx = this.accountTxRepository
                .findLastCommittedTransaction(branchNumber, accuntNumber).orElse(null);
        assertNull(lastCommittedTx);

        // commit
        initialEntity.commit(null);
        this.accountTxRepository.save(initialEntity);

        // after commit the transaction
        final AccountTx acceptedEntity = this.accountTxRepository.findById(txId).get();
        assertNotNull(acceptedEntity);
        assertEquals(TxStatus.ACCEPTED, acceptedEntity.getTxStatus());
        assertEquals(1, acceptedEntity.getTxSequence());
        assertEquals(acceptedEntity.getAmount(), acceptedEntity.getNewBalance());

        final AccountTx firstUncommittedTx2 = this.txOperations.execute(tx -> this.accountTxRepository
                .findUncommittedTransactions(branchNumber, accuntNumber, ZonedDateTime.now()).findFirst().orElse(null));
        assertNull(firstUncommittedTx2);

        final AccountTx lastCommittedTx2 = this.accountTxRepository
                .findLastCommittedTransaction(branchNumber, accuntNumber).orElse(null);
        assertNotNull(lastCommittedTx2);
        assertEquals(acceptedEntity, lastCommittedTx2);
    }

    @Test
    public void testInsertNewEntity_Unacceptable() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs().negate();

        final UUID txId = this.accountTxRepository
                .save(AccountTx.newEntity(branchNumber, accuntNumber, amount)).getTxId();
        final AccountTx initialEntity = this.accountTxRepository.findById(txId).get();

        initialEntity.commit(null);
        this.accountTxRepository.save(initialEntity);

        final AccountTx rejectedEntity = this.accountTxRepository.findById(txId).get();
        assertNotNull(rejectedEntity);
        assertEquals(TxStatus.REJECTED, rejectedEntity.getTxStatus());
        assertEquals(1, rejectedEntity.getTxSequence());
        assertEquals(new BigDecimal("0.00"), rejectedEntity.getNewBalance());

        final AccountTx lastCommittedTx = this.accountTxRepository
                .findLastCommittedTransaction(branchNumber, accuntNumber).orElse(null);
        assertNotNull(lastCommittedTx);
        assertEquals(rejectedEntity, lastCommittedTx);
    }
}
