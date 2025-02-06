package com.mahitotsu.brontes.api.repository;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.r2dbc.BadSqlGrammarException;

import com.mahitotsu.brontes.api.AbstractSpringTest;
import com.mahitotsu.brontes.api.entity.Account;

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
    public void testCreateNewAccount_Block() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();

        final Account account = this.accountRepository.createAccount(branchNumber, accountNumber).block();
        assertNotNull(account.getId());
        assertEquals(branchNumber, account.getBranchNumber());
        assertEquals(accountNumber, account.getAccountNumber());
        assertEquals(0L, account.getBalance());
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
    public void testCreateNewAccount_NullPK() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();

        StepVerifier.create(this.accountRepository.createAccount(branchNumber, null))
                .verifyError(DataIntegrityViolationException.class);
        StepVerifier.create(this.accountRepository.createAccount(null, accountNumber))
                .verifyError(DataIntegrityViolationException.class);
        StepVerifier.create(this.accountRepository.createAccount(null, null))
                .verifyError(DataIntegrityViolationException.class);
    }

    @Test
    public void testCreateNewAccount_InvalidFormatBranchNumber() {

        final String accountNumber = this.randomAccountNumber();

        StepVerifier.create(this.accountRepository.createAccount("01", accountNumber))
                .verifyError(DataIntegrityViolationException.class);
        StepVerifier.create(this.accountRepository.createAccount("0123", accountNumber))
                .verifyError(BadSqlGrammarException.class);
        StepVerifier.create(this.accountRepository.createAccount("01A", accountNumber))
                .verifyError(DataIntegrityViolationException.class);
    }

    @Test
    public void testCreateNewAccount_InvalidFormatAccountNumber() {

        final String branchNumber = this.randomBranchNumber();

        StepVerifier.create(this.accountRepository.createAccount(branchNumber, "012345"))
                .verifyError(DataIntegrityViolationException.class);
        StepVerifier.create(this.accountRepository.createAccount(branchNumber, "012345678"))
                .verifyError(BadSqlGrammarException.class);
        StepVerifier.create(this.accountRepository.createAccount(branchNumber, "012345B"))
                .verifyError(DataIntegrityViolationException.class);
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
        final Account account = this.accountRepository.updateBalance(branchNumber, accountNumber, amount3 * -1).block();
        assertEquals(amount1 + amount2 - amount3, account.getBalance());
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

        final int amount2 = (amount1 + 1) * -1;
        StepVerifier.create(this.accountRepository.updateBalance(branchNumber, accountNumber, amount2))
                .verifyError(DataIntegrityViolationException.class);
    }

    @Test
    public void testUpdateBalance_NotExistAccount() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        final int amount = SEED.nextInt(100);

        StepVerifier.create(this.accountRepository.updateBalance(branchNumber, accountNumber, amount))
                .expectNextCount(0)
                .verifyComplete();
        StepVerifier.create(this.accountRepository.updateBalance(branchNumber, accountNumber, amount))
                .verifyComplete();

        final Account account = this.accountRepository.updateBalance(branchNumber, accountNumber, amount).block();
        assertNull(account);
    }
}
