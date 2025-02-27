package com.mahitotsu.brontes.api.service;

import java.time.ZonedDateTime;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;

import com.mahitotsu.brontes.api.entity.AccountTx;

import jakarta.validation.constraints.Min;

@Service
@Validated
public class AccountTxService {

    @Autowired
    private AccountTxOperations accountTxOperations;

    public boolean openAccount(final String branchCode, final String accountNumber) {

        if (this.accountTxOperations.existsAccount(branchCode, accountNumber)) {
            return false;
        }

        final UUID txId = this.accountTxOperations.registerAccountTx(branchCode, accountNumber, 0);
        this.accountTxOperations.commitAccountTx(txId);

        return true;
    }

    public void closeAccount(final String branchCode, final String accountNumber) {
        // This method is not yet implemented.
    }

    public Long deposit(final String branchCode, final String accountNumber, @Min(0) final long amount) {
        return this.tx(branchCode, accountNumber, amount);
    }

    public Long withdraw(final String branchCode, final String accountNumber, @Min(0) final long amount) {
        return this.tx(branchCode, accountNumber, amount * -1);
    }

    private Long tx(final String branchCode, final String accountNumber, final long amount) {

        final UUID txId = this.accountTxOperations.registerAccountTx(branchCode, accountNumber, amount);

        AccountTx atx = null;
        while (atx == null) {
            try {
                atx = this.accountTxOperations.commitAccountTx(txId);
            } catch (CannotAcquireLockException e) {
                continue;
            }
        }

        switch (atx.getTxStatus()) {
            case ACCEPTED:
                return atx.getNewBalance().longValue();
            case REJECTED:
                throw new AccountTxRejectedException("This transaction is rejected.");
            default:
                throw new IllegalStateException("An unexpected situation has occurred.");
        }
    }

    public Long getBalance(final String branchCode, final String accountNumber) {
        return this.accountTxOperations.getLastBalance(branchCode, accountNumber, ZonedDateTime.now());
    }
}
