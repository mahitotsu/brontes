package com.mahitotsu.brontes.api.repository;

import static org.springframework.data.relational.core.query.Criteria.*;
import static org.springframework.data.relational.core.query.Query.*;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.r2dbc.core.R2dbcEntityOperations;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import com.mahitotsu.brontes.api.entity.Account;

import reactor.core.publisher.Mono;

@Repository
public class AccountRepository {

    @Autowired
    private R2dbcEntityOperations operations;

    @Transactional
    public Mono<Account> openNewAccount(final String branchNumber, final String accountNumber) {

        return this.getAccount(branchNumber, accountNumber)
                .switchIfEmpty(Mono.defer(() -> {
                    final Account newAccount = new Account();
                    newAccount.setBranchNumber(branchNumber);
                    newAccount.setAccountNumber(accountNumber);
                    newAccount.setBalance(0L);
                    return this.operations.insert(newAccount).then(this.getAccount(branchNumber, accountNumber));
                }));
    }

    @Transactional
    public Mono<Account> updateBalance(final String branchNumber, final String accountNumber, final long amount) {

        return this.getAccount(branchNumber, accountNumber)
                .flatMap(account -> {
                    account.setBalance(account.getBalance() + amount);
                    return this.operations.update(account);
                });
    }

    @Transactional(readOnly = true)
    public Mono<Account> getAccount(final String branchNumber, final String accountNumber) {

        if (branchNumber == null || accountNumber == null) {
            return Mono.empty();
        }

        return this.operations.selectOne(query(
                where("branchNumber").is(branchNumber)
                        .and(where("accountNumber").is(accountNumber))),
                Account.class);
    }
}
