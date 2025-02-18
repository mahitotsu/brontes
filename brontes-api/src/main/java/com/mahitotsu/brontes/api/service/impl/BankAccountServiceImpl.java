package com.mahitotsu.brontes.api.service.impl;

import java.math.BigDecimal;
import java.util.Map;
import java.util.UUID;

import org.springframework.stereotype.Service;

import com.mahitotsu.brontes.api.model.AccountTransaction;
import com.mahitotsu.brontes.api.repository.AccountTransactionRepository;
import com.mahitotsu.brontes.api.service.BankAccountService;

import reactor.core.publisher.Mono;

@Service
public class BankAccountServiceImpl extends AbstractIdempotentService implements BankAccountService {

    private AccountTransactionRepository accountTransactionRepository;

    @Override
    public Mono<Boolean> openAccount(final UUID idemKey, final String branchNumber, final String accountNumber) {

        final Map<String, Object> payload = this.newMapBuilder()
                .put("branchNumber", branchNumber)
                .put("accountNumber", accountNumber)
                .build();

        return this.executeWithIdempotencyMono(idemKey, payload, () -> {
            final AccountTransaction entity = new AccountTransaction();
            entity.setBranchNUmber(branchNumber);
            entity.setAccountNumber(accountNumber);
            return this.accountTransactionRepository.save(entity);
        }).then(Mono.just(true));
    }

    @Override
    public Mono<Boolean> closeAccount(final UUID idemKey, final String branchNumber, final String accountNumber) {
        return Mono.just(true);
    }

    @Override
    public Mono<BigDecimal> deposit(final UUID idemKey, final String branchNumber, final String accountNumber,
            final BigDecimal amount) {
        return null;
    }

    @Override
    public Mono<BigDecimal> withdraw(final UUID idemKey, final String branchNumber, final String accountNumber,
            final BigDecimal amount) {
        return null;
    }
}
