package com.mahitotsu.brontes.api.repository;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.dao.DuplicateKeyException;

import com.mahitotsu.brontes.api.AbstractSpringTest;

import reactor.test.StepVerifier;

public class PointEventRepositoryTest extends AbstractSpringTest {

    private static final Random SEED = new Random();

    @Autowired
    private AccountRepository accountRepository;

    private String randomBranchNumber() {
        return String.format("%03d", SEED.nextInt(100));
    }

    private String randomAccountNumber() {
        return String.format("%07d", SEED.nextInt(10000000));
    }

    @Test
    public void testCreateNewAccount() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();

        StepVerifier.create(this.accountRepository.createAccount(branchNumber, accountNumber))
                .assertNext(account -> {
                    assertNotNull(account.getId());
                    assertEquals(branchNumber, account.getBranchNumber());
                    assertEquals(accountNumber, account.getAccountNumber());
                    assertEquals(0L, account.getBalance());
                })
                .verifyComplete();
    }

    @Test
    public void testCreateNewAccount_DuplicateUniqueKey() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();

        StepVerifier.create(this.accountRepository.createAccount(branchNumber, accountNumber))
                .expectNextCount(1)
                .verifyComplete();

        StepVerifier.create(this.accountRepository.createAccount(branchNumber, accountNumber))
                .verifyError(DuplicateKeyException.class);
    }

    @Test
    public void testUpdateBalance() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        final int amount1 = SEED.nextInt(100);

        StepVerifier.create(this.accountRepository.createAccount(branchNumber, accountNumber))
                .expectNextCount(1)
                .verifyComplete();

        StepVerifier.create(this.accountRepository.updateBalance(branchNumber, accountNumber, amount1))
                .assertNext(account -> {
                    assertEquals(amount1, account.getBalance());
                })
                .verifyComplete();

        final int amount2 = SEED.nextInt(100);
        StepVerifier.create(this.accountRepository.updateBalance(branchNumber, accountNumber, amount2))
                .assertNext(account -> {
                    assertEquals(amount1 + amount2, account.getBalance());
                })
                .verifyComplete();

        final int amount3 = SEED.nextInt(amount1 + amount2);
        StepVerifier.create(this.accountRepository.updateBalance(branchNumber, accountNumber, amount3 * -1))
                .assertNext(account -> {
                    assertEquals(amount1 + amount2 - amount3, account.getBalance());
                })
                .verifyComplete();
    }

    @Test
    public void testUpdateBalance_NegativeBalance() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        final int amount1 = SEED.nextInt(100);

        StepVerifier.create(this.accountRepository.createAccount(branchNumber, accountNumber))
                .expectNextCount(1)
                .verifyComplete();

        StepVerifier.create(this.accountRepository.updateBalance(branchNumber, accountNumber, amount1))
                .expectNextCount(1)
                .verifyComplete();

        final int amount2 = (amount1 + SEED.nextInt(10)) * -1;
        StepVerifier.create(this.accountRepository.updateBalance(branchNumber, accountNumber, amount2))
                .verifyError(DataIntegrityViolationException.class);
    }
}
