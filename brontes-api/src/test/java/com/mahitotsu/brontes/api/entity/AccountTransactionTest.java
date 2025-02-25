package com.mahitotsu.brontes.api.entity;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.List;
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

        final AccountTransaction newEntity = AccountTransaction.newEntity(branchNumber, accuntNumber, amount);
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
                tx -> this.entityManager.persist(AccountTransaction.newEntity(branchNumber, accuntNumber, amount))));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity_InvalidLengthBranchNumber() {

        final Integer branchNumber = this.randomBranchNumber() + 1000;
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.txOperations.executeWithoutResult(
                tx -> this.entityManager.persist(AccountTransaction.newEntity(branchNumber, accuntNumber, amount))));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity_NullAccountNumber() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = null;
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.txOperations.executeWithoutResult(
                tx -> this.entityManager.persist(AccountTransaction.newEntity(branchNumber, accuntNumber, amount))));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity_InvalidLengthAccountNumber() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber() + 10000000;
        final BigDecimal amount = this.randomAmount().abs();

        assertThrows(DataIntegrityViolationException.class, () -> this.txOperations.executeWithoutResult(
                tx -> this.entityManager.persist(AccountTransaction.newEntity(branchNumber, accuntNumber, amount))));
    }

    @SuppressWarnings("unused")
    @Test
    public void testInsertNewEntity_Acceptable() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        final AccountTransaction newEntity = AccountTransaction.newEntity(branchNumber, accuntNumber, amount);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.persist(newEntity));
        final AccountTransaction initialEntity = this.entityManager.find(AccountTransaction.class, newEntity.getTxId());

        initialEntity.commit(null);
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
    public void testInsertNewEntity_Unacceptable() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs().negate();

        final AccountTransaction newEntity = AccountTransaction.newEntity(branchNumber, accuntNumber, amount);
        this.txOperations.executeWithoutResult(tx -> this.entityManager.persist(newEntity));
        final AccountTransaction initialEntity = this.entityManager.find(AccountTransaction.class, newEntity.getTxId());

        initialEntity.commit(null);
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
    public void testCommitTransactions() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accountNumber = this.randomAccountNumber();

        final List<Integer> amounts = Arrays.asList(100, 200, -50, 100, -300, -100, 100);
        amounts.forEach(amount -> this.txOperations.executeWithoutResult(tx -> this.entityManager
                .persist(AccountTransaction.newEntity(branchNumber, accountNumber, new BigDecimal(amount)))));

        final AccountTransaction lastEntity = this.txOperations.execute(tx -> AccountTransaction.commitTransactions(
                null,
                this.entityManager.createQuery("""
                        SELECT atx FROM AccountTransaction atx
                        WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber AND txSequence IS NULL
                        ORDER BY txTimestamp
                        """, AccountTransaction.class)
                        .setParameter("branchNumber", branchNumber)
                        .setParameter("accountNumber", accountNumber)
                        .getResultStream()));
        assertNotNull(lastEntity);
        assertEquals(TxStatus.ACCEPTED, lastEntity.getTxStatus());
        assertEquals(amounts.size(), lastEntity.getTxSequence());
        assertEquals(new BigDecimal("150.00"), lastEntity.getNewBalance());

        this.txOperations.executeWithoutResult(tx -> {
            final List<AccountTransaction> txList = this.entityManager.createQuery("""
                    SELECT atx FROM AccountTransaction atx
                    WHERE branchNumber = :branchNumber AND accountNumber = :accountNumber AND txSequence IS NOT NULL
                    ORDER BY txSequence
                    """, AccountTransaction.class)
                    .setParameter("branchNumber", branchNumber)
                    .setParameter("accountNumber", accountNumber)
                    .getResultList();
            assertEquals(amounts.size(), txList.size());

            final AccountTransaction rejectedTx = txList.get(5);
            assertEquals(TxStatus.REJECTED, rejectedTx.getTxStatus());
            assertEquals(6, rejectedTx.getTxSequence());
            assertEquals(new BigDecimal("50.00"), rejectedTx.getNewBalance());
            assertEquals(rejectedTx.getNewBalance(), txList.get(4).getNewBalance());
        });
    }
}
