package com.mahitotsu.brontes.api.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

import com.mahitotsu.brontes.api.AbstractTestBase;

import jakarta.validation.ConstraintViolationException;

public class AccountTxServiceTest extends AbstractTestBase {

    private static final Random RANDOM = new Random();

    @Autowired
    private AccountTxService accountTxService;

    private String randomBranchNumber() {
        return String.format("%03d", RANDOM.nextInt(1000));
    }

    private String randomAccountNumber() {
        return String.format("%07d", RANDOM.nextInt(10000000));
    }

    @Test
    public void testOpenAccount_InvalidBranchNumber() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();

        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.openAccount(branchNumber + "0", accountNumber));
        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.openAccount(branchNumber.substring(1), accountNumber));
        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.openAccount(branchNumber.substring(1) + "a", accountNumber));
    }

    @Test
    public void testOpenAccount_InvalidAccountNumber() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();

        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.openAccount(branchNumber, accountNumber + "0"));
        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.openAccount(branchNumber, accountNumber.substring(1)));
        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.openAccount(branchNumber, accountNumber.substring(1) + "a"));
    }

    @Test
    public void testOpenAccount() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();

        final boolean result1 = this.accountTxService.openAccount(branchNumber, accountNumber);
        assertTrue(result1);

        final boolean result2 = this.accountTxService.openAccount(branchNumber, accountNumber);
        assertFalse(result2);

        final long balance = this.accountTxService.getBalance(branchNumber, accountNumber);
        assertEquals(0, balance);
    }

    @Test
    public void testDeposit_NegativeAmount() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long amount = -100;
        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.deposit(branchNumber, accountNumber, amount));
    }

    @Test
    public void testDeposit_ExceededAmount() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long amount = 10000000000000L;
        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.deposit(branchNumber, accountNumber, amount));
    }

    @Test
    public void testDeposit_ExceededBalance() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long initialAmount = 10000000000000L - 1;
        this.accountTxService.deposit(branchNumber, accountNumber, initialAmount);

        final long amount = 1;
        assertThrows(AccountTxRejectedException.class,
                () -> this.accountTxService.deposit(branchNumber, accountNumber, amount));
    }

    @Test
    public void testWithdraw_NegativeAmount() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long amount = -100;
        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.withdraw(branchNumber, accountNumber, amount));
    }

    @Test
    public void testWithdraw_ExceededAmount() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long amount = 10000000000000L;
        assertThrows(ConstraintViolationException.class,
                () -> this.accountTxService.withdraw(branchNumber, accountNumber, amount));
    }

    @Test
    public void testDeposit() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long amount = 100;
        final Long newBalance = this.accountTxService.deposit(branchNumber, accountNumber, amount);
        assertNotNull(newBalance);
        assertEquals(amount, newBalance.longValue());
    }

    @Test
    public void testWithdraw() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long initialAmount = 1000;
        this.accountTxService.deposit(branchNumber, accountNumber, initialAmount);

        final long amount = 123;
        final Long newBalance = this.accountTxService.withdraw(branchNumber, accountNumber, amount);
        assertNotNull(newBalance);
        assertEquals(initialAmount - amount, newBalance.longValue());
    }

    @Test
    public void testWithdraw_NegativeBalance() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long initialAmount = 1000;
        this.accountTxService.deposit(branchNumber, accountNumber, initialAmount);

        final long amount = 1230;
        assertThrows(AccountTxRejectedException.class,
                () -> this.accountTxService.withdraw(branchNumber, accountNumber, amount));
    }

    @Test
    public void testDepositAndWithdraw() {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        this.accountTxService.openAccount(branchNumber, accountNumber);

        final long[] amounts = new long[] { 123, 456, 789, -321, -654, 987 };
        final long[] expecteds = new long[amounts.length];
        for (int i = 0; i < amounts.length; i++) {
            expecteds[i] = (i > 0 ? expecteds[i - 1] + amounts[i] : amounts[i]);
        }

        for (int i = 0; i < amounts.length; i++) {
            final Long newBalance = amounts[i] >= 0
                    ? this.accountTxService.deposit(branchNumber, accountNumber, amounts[i])
                    : this.accountTxService.withdraw(branchNumber, accountNumber, amounts[i] * -1);
            assertNotNull(newBalance);
            assertEquals(expecteds[i], newBalance.longValue());
        }
    }

    @Test
    public void testDepositAndWithdraw_SingleThread() throws Exception {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        final long initialBalance = 10000;
        this.accountTxService.openAccount(branchNumber, accountNumber);
        this.accountTxService.deposit(branchNumber, accountNumber, initialBalance);

        final long[] amounts = new long[] { 123, 456, 789, -321, -654, 987 };
        final long[] expecteds = new long[amounts.length];
        for (int i = 0; i < amounts.length; i++) {
            expecteds[i] = (i > 0 ? expecteds[i - 1] + amounts[i] : initialBalance + amounts[i]);
        }

        final List<Callable<Long>> taskList = new ArrayList<>(amounts.length);
        for (int i = 0; i < amounts.length; i++) {
            final long amount = amounts[i];
            taskList.add(() -> amount >= 0
                    ? this.accountTxService.deposit(branchNumber, accountNumber, amount)
                    : this.accountTxService.withdraw(branchNumber, accountNumber, amount * -1));
        }

        final ExecutorService executor = Executors.newSingleThreadExecutor();
        final List<Future<Long>> results = executor.invokeAll(taskList);
        for (int i = 0; i < amounts.length; i++) {
            final Future<Long> result = results.get(i);
            assertTrue(result.isDone());
            assertNotNull(result.get());
        }

        final Long balance = this.accountTxService.getBalance(branchNumber, accountNumber);
        assertEquals(expecteds[expecteds.length - 1], balance.longValue());
    }

    @Test
    public void testDepositAndWithdraw_MultiThreads() throws Exception {

        final String branchNumber = this.randomBranchNumber();
        final String accountNumber = this.randomAccountNumber();
        final long initialBalance = 10000;
        this.accountTxService.openAccount(branchNumber, accountNumber);
        this.accountTxService.deposit(branchNumber, accountNumber, initialBalance);

        final long[] amounts = new long[] { 123, 456, 789, -321, -654, 987 };
        final long[] expecteds = new long[amounts.length];
        for (int i = 0; i < amounts.length; i++) {
            expecteds[i] = (i > 0 ? expecteds[i - 1] + amounts[i] : initialBalance + amounts[i]);
        }

        final List<Callable<Long>> taskList = new ArrayList<>(amounts.length);
        for (int i = 0; i < amounts.length; i++) {
            final long amount = amounts[i];
            taskList.add(() -> amount >= 0
                    ? this.accountTxService.deposit(branchNumber, accountNumber, amount)
                    : this.accountTxService.withdraw(branchNumber, accountNumber, amount * -1));
        }

        final ExecutorService executor = Executors.newFixedThreadPool(3);
        final List<Future<Long>> results = executor.invokeAll(taskList);
        for (int i = 0; i < amounts.length; i++) {
            final Future<Long> result = results.get(i);
            assertTrue(result.isDone());
            assertNotNull(result.get());
        }

        final Long balance = this.accountTxService.getBalance(branchNumber, accountNumber);
        assertEquals(expecteds[expecteds.length - 1], balance.longValue());
    }
}
