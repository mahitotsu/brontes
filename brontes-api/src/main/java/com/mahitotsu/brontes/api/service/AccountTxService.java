package com.mahitotsu.brontes.api.service;

import java.time.ZonedDateTime;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;

import com.mahitotsu.brontes.api.entity.AccountTx;

@Service
@Validated
public class AccountTxService {

    @Autowired
    private AccountTxOperations accountTxOperations;

    public boolean openAccount(final String branchCode, final String accountNumber) {

        if (this.accountTxOperations.existsAccount(branchCode, accountNumber)) {
            return false;
        }

        final AccountTx entity = this.accountTxOperations.registerAccountTx(branchCode, accountNumber, 0);
        this.accountTxOperations.commitAccountTx(entity.getTxId());

        return true;
    }

    public void closeAccount(final String branchCode, final String accountNumber) {
        // This method is not yet implemented.
    }

    public Integer deposit(final String branchCode, final String accountNumber, final int amount) {
        return this.tx(branchCode, accountNumber, amount);
    }

    public Integer withdraw(final String branchCode, final String accountNumber, final int amount) {
        return this.tx(branchCode, accountNumber, amount * -1);
    }

    private Integer tx(final String branchCode, final String accountNumber, final int amount) {

        final AccountTx entity = this.accountTxOperations.registerAccountTx(branchCode, accountNumber, amount);
        this.accountTxOperations.commitAccountTx(entity.getTxId());

        final Integer balance = this.accountTxOperations.regirievePastBalance(entity.getTxId());
        if (balance == null) {
            throw new IllegalStateException("An unexpected situation has occurred.");
        }
        return balance;
    }

    public Integer getBalance(final String branchCode, final String accountNumber) {
        return this.accountTxOperations.getLastBalance(branchCode, accountNumber, ZonedDateTime.now());
    }
}
