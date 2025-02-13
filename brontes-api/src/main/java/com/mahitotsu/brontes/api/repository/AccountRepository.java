package com.mahitotsu.brontes.api.repository;

import static org.springframework.data.relational.core.query.Criteria.*;
import static org.springframework.data.relational.core.query.Query.*;
import static org.springframework.data.relational.core.query.Update.*;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.r2dbc.core.R2dbcEntityOperations;
import org.springframework.data.relational.core.query.Query;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import com.mahitotsu.brontes.api.entity.Account;

import reactor.core.publisher.Mono;

@Repository
public class AccountRepository {

    @Autowired
    private R2dbcEntityOperations operations;

    @Transactional
    public Mono<Account> openAccount(final String branchNumber, final String accountNumber) {

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
                    final long newBalance = account.getBalance() + amount;
                    account.setBalance(newBalance);
                    return this.operations.update(Account.class).matching(this.buildQuery(branchNumber, accountNumber))
                            .apply(update("balance", newBalance)).thenReturn(account);
                });
    }

    @Transactional(readOnly = true)
    public Mono<Account> getAccount(final String branchNumber, final String accountNumber) {

        return this.operations.selectOne(this.buildQuery(branchNumber, accountNumber), Account.class);
    }

    @Transactional
    public Mono<Boolean> closeAccount(final String branchNumber, final String accountNumber) {

        return this.operations.delete(Account.class).matching(this.buildQuery(branchNumber, accountNumber)).all()
                .map(count -> count > 0);
    }

    private Query buildQuery(final String branchNumber, final String accountNumber) {

        return query(branchNumber == null ? where("branchNumber").isNull()
                : where("branchNumber").is(branchNumber).and(accountNumber == null ? where("accountNumber").isNull()
                        : where("accountNumber").is(accountNumber)));
    }
}
