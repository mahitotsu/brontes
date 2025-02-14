package com.mahitotsu.brontes.api.repository;

import static org.springframework.data.relational.core.query.Criteria.*;
import static org.springframework.data.relational.core.query.Query.*;
import static org.springframework.data.relational.core.query.Update.*;

import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.r2dbc.core.R2dbcEntityOperations;
import org.springframework.data.relational.core.query.Query;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import com.mahitotsu.brontes.api.entity.Account;
import com.mahitotsu.brontes.api.entity.Transaction;
import com.mahitotsu.brontes.api.entity.TransactionType;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@Repository
public class AccountRepository {

    @Autowired
    private R2dbcEntityOperations operations;

    @Transactional
    public Mono<Account> openAccount(final String branchNumber, final String accountNumber) {

        final UUID txId = UUID.randomUUID();
        return this.getAccount(branchNumber, accountNumber)
                .switchIfEmpty(Mono.defer(() -> {

                    final Account newAccount = new Account();
                    newAccount.setBranchNumber(branchNumber);
                    newAccount.setAccountNumber(accountNumber);
                    newAccount.setBalance(0L);

                    final Transaction transaction = new Transaction();
                    transaction.setTxId(txId);
                    transaction.setTxType(TransactionType.O);
                    transaction.setBranchNumber(branchNumber);
                    transaction.setAccountNumber(accountNumber);
                    transaction.setAmount(0L);

                    return Mono.zip(this.operations.insert(newAccount), this.operations.insert(transaction))
                            .map(tuple -> tuple.getT1());
                }));
    }

    @Transactional
    public Mono<Account> updateBalance(final String branchNumber, final String accountNumber, final long amount) {

        return this.getAccount(branchNumber, accountNumber)
                .flatMap(account -> {
                    final long newBalance = account.getBalance() + amount;
                    account.setBalance(newBalance);
                    return this.operations.update(Account.class).matching(this.buildQuery(branchNumber, accountNumber))
                            .apply(update("balance", newBalance)).then(this.getAccount(branchNumber, accountNumber));
                });
    }

    @Transactional(readOnly = true)
    public Mono<Account> getAccount(final String branchNumber, final String accountNumber) {
        return this.operations.selectOne(this.buildQuery(branchNumber, accountNumber), Account.class);
    }

    @Transactional(readOnly = true)
    public Flux<Transaction> listTransactions(final String branchNumber, final String accountNumber) {
        return this.operations.select(Transaction.class).matching(this.buildQuery(branchNumber, accountNumber)).all();
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
