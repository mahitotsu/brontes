package com.mahitotsu.brontes.api.service;

import java.math.BigDecimal;
import java.util.UUID;

import reactor.core.publisher.Mono;

public interface BankAccountService {

    Mono<Boolean> openAccount(UUID idemKey, String branchNumber, String accountNumber);

    Mono<Boolean> closeAccount(UUID idemKey, String branchNumber, String accountNumber);

    Mono<BigDecimal> deposit(UUID idemKey, String branchNumber, String accountNumber, BigDecimal amount);

    Mono<BigDecimal> withdraw(UUID idemKey, String branchNumber, String accountNumber, BigDecimal amount);
}
