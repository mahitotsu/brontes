package com.mahitotsu.brontes.api.service.impl;

import java.math.BigDecimal;
import java.util.Map;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.mahitotsu.brontes.api.model.AccountTransaction;
import com.mahitotsu.brontes.api.model.AccountTransaction.TxType;
import com.mahitotsu.brontes.api.repository.AccountTransactionRepository;
import com.mahitotsu.brontes.api.service.BankAccountService;

import reactor.core.publisher.Mono;

@Service
public class BankAccountServiceImpl extends AbstractIdempotentService implements BankAccountService {

    @Autowired
    private AccountTransactionRepository accountTransactionRepository;

    @Override
    public Mono<Boolean> open(final UUID idemKey, final String branchNumber, final String accountNumber) {

        final Map<String, Object> payload = this.newMapBuilder()
                .put("branchNumber", branchNumber)
                .put("accountNumber", accountNumber)
                .build();

        final AccountTransaction entity = new AccountTransaction();
        entity.setTxSeq(0L);
        entity.setTxType(TxType.O);
        entity.setBranchNumber(branchNumber);
        entity.setAccountNumber(accountNumber);

        return this.executeWithIdempotencyMono(idemKey, payload, this.accountTransactionRepository.save(entity))
                .thenReturn(true);
    }

    @Override
    public Mono<Boolean> close(final UUID idemKey, final String branchNumber, final String accountNumber) {
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

    public Mono<BigDecimal> queryBalance(final String branchNumber, final String accountNumber) {
        return this.queryMono(this.accountTransactionRepository.queryBalance(branchNumber, accountNumber));
    }
}
