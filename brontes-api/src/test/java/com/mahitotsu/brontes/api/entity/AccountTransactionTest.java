package com.mahitotsu.brontes.api.entity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.math.BigDecimal;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.transaction.support.TransactionOperations;

import com.mahitotsu.brontes.api.AbstractTestBase;
import com.mahitotsu.brontes.api.entity.AccountTransaction.TxStatus;

import jakarta.persistence.EntityManager;

public class AccountTransactionTest extends AbstractTestBase {

    private static final Random RANDOM = new Random();

    @Autowired
    private EntityManager entityManager;

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

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        final AccountTransaction newEntity = new AccountTransaction(branchNumber, accuntNumber, amount);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.persist(newEntity));
        assertNotNull(newEntity.getTxId());

        final AccountTransaction savedEnttiy = this.entityManager.find(AccountTransaction.class, newEntity.getTxId());
        assertNotNull(savedEnttiy.getTxTimestamp());
        assertEquals(AccountTransaction.TxStatus.REGISTERED, savedEnttiy.getTxStatus());

        assertNull(savedEnttiy.getTxSequence());
        assertNull(savedEnttiy.getNewBalance());
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity_NullBranchNumber() {

        final Integer branchNumber = null;
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.txOperations.executeWithoutResult(
                tx -> this.entityManager.persist(new AccountTransaction(branchNumber, accuntNumber, amount))));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity_InvalidLengthBranchNumber() {

        final Integer branchNumber = this.randomBranchNumber() + 1000;
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.txOperations.executeWithoutResult(
                tx -> this.entityManager.persist(new AccountTransaction(branchNumber, accuntNumber, amount))));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity_NullAccountNumber() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = null;
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.txOperations.executeWithoutResult(
                tx -> this.entityManager.persist(new AccountTransaction(branchNumber, accuntNumber, amount))));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity_InvalidLengthAccountNumber() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber() + 10000000;
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.txOperations.executeWithoutResult(
                tx -> this.entityManager.persist(new AccountTransaction(branchNumber, accuntNumber, amount))));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntityAndAccept() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        final AccountTransaction newEntity = new AccountTransaction(branchNumber, accuntNumber, amount);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.persist(newEntity));
        final AccountTransaction initialEntity = this.entityManager.find(AccountTransaction.class, newEntity.getTxId());

        initialEntity.accept(null);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.merge(initialEntity));
        final AccountTransaction acceptedEntity = this.entityManager.find(AccountTransaction.class,
                newEntity.getTxId());

        assertNotNull(acceptedEntity);
        assertEquals(TxStatus.ACCEPTED, acceptedEntity.getTxStatus());
        assertEquals(1, acceptedEntity.getTxSequence());
        assertEquals(acceptedEntity.getAmount(), acceptedEntity.getNewBalance());
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntityAndReject() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        final AccountTransaction newEntity = new AccountTransaction(branchNumber, accuntNumber, amount);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.persist(newEntity));
        final AccountTransaction initialEntity = this.entityManager.find(AccountTransaction.class, newEntity.getTxId());

        initialEntity.reject(null);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.merge(initialEntity));
        final AccountTransaction rejectedEntity = this.entityManager.find(AccountTransaction.class,
                newEntity.getTxId());

        assertNotNull(rejectedEntity);
        assertEquals(TxStatus.REJECTED, rejectedEntity.getTxStatus());
        assertEquals(1, rejectedEntity.getTxSequence());
        assertEquals(new BigDecimal("0.00"), rejectedEntity.getNewBalance());
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntityAndAccept_NegativeAmount() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs().negate();

        final AccountTransaction newEntity = new AccountTransaction(branchNumber, accuntNumber, amount);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.persist(newEntity));
        final AccountTransaction initialEntity = this.entityManager.find(AccountTransaction.class, newEntity.getTxId());

        initialEntity.accept(null);
        assertThrows(DataIntegrityViolationException.class, () -> this.txOperations
                .executeWithoutResult(tx -> this.entityManager.merge(initialEntity)));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntityAndReject_NegativeAmount() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs().negate();

        final AccountTransaction newEntity = new AccountTransaction(branchNumber, accuntNumber, amount);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.persist(newEntity));
        final AccountTransaction initialEntity = this.entityManager.find(AccountTransaction.class, newEntity.getTxId());

        initialEntity.reject(null);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.merge(initialEntity));
        final AccountTransaction rejectedEntity = this.entityManager.find(AccountTransaction.class,
                newEntity.getTxId());

        assertEquals(new BigDecimal("0.00"), rejectedEntity.getNewBalance());
    }
}
