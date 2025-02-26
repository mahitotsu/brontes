package com.mahitotsu.brontes.api.repository;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.stream.Collectors;

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
    public void testInsertNewEntity_Acceptable() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accuntNumber = this.randomAccountNumber();
        final BigDecimal amount = this.randomAmount().abs();

        final UUID txId = this.accountTxRepository
                .save(AccountTx.newEntity(branchNumber, accuntNumber, amount)).getTxId();
        final AccountTx initialEntity = this.accountTxRepository.findById(txId).get();

        initialEntity.commit(null);
        this.accountTxRepository.save(initialEntity);

        final AccountTx acceptedEntity = this.accountTxRepository.findById(txId).get();
        assertNotNull(acceptedEntity);
        assertEquals(TxStatus.ACCEPTED, acceptedEntity.getTxStatus());
        assertEquals(1, acceptedEntity.getTxSequence());
        assertEquals(acceptedEntity.getAmount(), acceptedEntity.getNewBalance());
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
    }

    @SuppressWarnings("unused")
    @Test
    public void testCommitTransactions() {

        final Integer branchNumber = this.randomBranchNumber();
        final Integer accountNumber = this.randomAccountNumber();

        final List<Integer> amounts = Arrays.asList(100, 200, -50, 100, -300, -100, 100);
        this.accountTxRepository.saveAll(
                amounts.stream().map(a -> AccountTx.newEntity(branchNumber, accountNumber, new BigDecimal(a)))
                        .collect(Collectors.toList()));

        final AccountTx lastEntity = this.txOperations.execute(tx -> AccountTx.commitTransactions(
                null, this.accountTxRepository.findAllUncommittedTransactions(branchNumber, accountNumber,
                        ZonedDateTime.now().plusDays(1))));
        assertNotNull(lastEntity);
        assertEquals(TxStatus.ACCEPTED, lastEntity.getTxStatus());
        assertEquals(amounts.size(), lastEntity.getTxSequence());
        assertEquals(new BigDecimal("150.00"), lastEntity.getNewBalance());

        this.txOperations.executeWithoutResult(tx -> {
            final List<AccountTx> txList = this.accountTxRepository
                    .findAllCommittedTransactions(branchNumber, accountNumber, lastEntity.getTxTimestamp())
                    .collect(Collectors.toList());
            assertEquals(amounts.size(), txList.size());

            final AccountTx rejectedTx = txList.get(5);
            assertEquals(TxStatus.REJECTED, rejectedTx.getTxStatus());
            assertEquals(6, rejectedTx.getTxSequence());
            assertEquals(new BigDecimal("50.00"), rejectedTx.getNewBalance());
            assertEquals(rejectedTx.getNewBalance(), txList.get(4).getNewBalance());
        });

        final AccountTx lastEntity2 = this.accountTxRepository
                .findOneLastCommittedTransaction(branchNumber, accountNumber).get();
        assertEquals(lastEntity, lastEntity2);
    }
}
