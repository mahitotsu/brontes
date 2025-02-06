package com.mahitotsu.brontes.api.repository;

import static org.springframework.data.relational.core.query.Criteria.where;
import static org.springframework.data.relational.core.query.Query.query;

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
    public Mono<Account> createAccount(final String branchNumber, final String accountNumber) {

        final Account newEntity = new Account();
        newEntity.setBranchNumber(branchNumber);
        newEntity.setAccountNumber(accountNumber);
        newEntity.setBalance(0L);

        return this.operations.insert(Account.class).using(newEntity);
    }

    @Transactional
    public Mono<Account> updateBalance(final String branchNumber, final String accountNumber, final long amount) {

        return this.operations.selectOne(query(
                where("branchNumber").is(branchNumber)
                        .and(where("accountNumber").is(accountNumber))),
                Account.class)
                .flatMap(account -> {
                    account.setBalance(account.getBalance() + amount);
                    return this.operations.update(account);
                });
    }
}
