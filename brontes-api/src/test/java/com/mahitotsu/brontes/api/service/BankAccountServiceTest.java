package com.mahitotsu.brontes.api.service;

import java.math.BigDecimal;
import java.util.Random;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

import com.mahitotsu.brontes.api.AbstractSpringTest;

import reactor.test.StepVerifier;

public class BankAccountServiceTest extends AbstractSpringTest {

    private static final Random RANDOM = new Random();

    private static String randomBranchNumber() {
        return String.format("%03d", RANDOM.nextInt(1000));
    }

    private static String randomAccountNumber() {
        return String.format("%07d", RANDOM.nextInt(10000000));
    }

    @Autowired
    private BankAccountService bankAccountService;

    @Test
    public void testOpenAccount() {

        final UUID idemKey = UUID.randomUUID();
        final String branchNumber = randomBranchNumber();
        final String accuntNuString = randomAccountNumber();

        StepVerifier.create(this.bankAccountService.open(idemKey, branchNumber, accuntNuString))
                .expectNext(true)
                .verifyComplete();

        StepVerifier.create(this.bankAccountService.queryBalance(branchNumber, accuntNuString))
                .expectNext(new BigDecimal("0.00"))
                .verifyComplete();
    }
}
