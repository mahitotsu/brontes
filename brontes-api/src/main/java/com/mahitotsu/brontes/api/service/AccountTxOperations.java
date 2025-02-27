package com.mahitotsu.brontes.api.service;

import java.math.BigDecimal;
import java.time.ZonedDateTime;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;

import com.mahitotsu.brontes.api.entity.AccountTx;
import com.mahitotsu.brontes.api.entity.AccountTx.TxStatus;
import com.mahitotsu.brontes.api.repository.AccountTxRepository;

import jakarta.validation.constraints.Digits;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

@Component
@Validated
public class AccountTxOperations {

    @Autowired
    private AccountTxRepository accountTxRepository;

    @Transactional(readOnly = true)
    public boolean existsAccount(
            @NotNull @Pattern(regexp = "^[0-9]{3}$") final String branchNumber,
            @NotNull @Pattern(regexp = "^[0-9]{7}$") final String accountNumber) {

        return this.accountTxRepository.findLastCommittedTransaction(Integer.valueOf(branchNumber),
                Integer.valueOf(accountNumber)).orElse(null) != null;
    }

    @Transactional(readOnly = true)
    public Long getLastBalance(
            @NotNull @Pattern(regexp = "^[0-9]{3}$") final String branchNumber,
            @NotNull @Pattern(regexp = "^[0-9]{7}$") final String accountNumber,
            final ZonedDateTime txTimestamp) {

        return this.accountTxRepository.findLastCommittedTransaction(Integer.valueOf(branchNumber),
                Integer.valueOf(accountNumber)).map(entity -> entity.getNewBalance().longValue())
                .orElse(null);
    }

    @Transactional
    public UUID registerAccountTx(
            @NotNull @Pattern(regexp = "^[0-9]{3}$") final String branchNumber,
            @NotNull @Pattern(regexp = "^[0-9]{7}$") final String accountNumber,
            @Digits(integer = 13, fraction = 2) final long amount) {
        return this.accountTxRepository
                .save(AccountTx.newEntity(Integer.valueOf(branchNumber),
                        Integer.valueOf(accountNumber), new BigDecimal(amount)))
                .getTxId();
    }

    @Transactional
    public AccountTx commitAccountTx(@NotNull final UUID txId) {

        final AccountTx entity = this.accountTxRepository.findById(txId).get();
        if (entity.getTxStatus() != TxStatus.REGISTERED) {
            return entity;
        }

        final Integer branchNumber = entity.getBranchNumber();
        final Integer accountNumber = entity.getAccountNumber();

        final AccountTx lastCommittedTx = this.accountTxRepository
                .findLastCommittedTransaction(branchNumber, accountNumber).orElse(null);
        final AccountTx firstUncommittedTx = this.accountTxRepository
                .findUncommittedTransactions(branchNumber, accountNumber, entity.getTxTimestamp())
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("An unexpected situation has occurred."));

        firstUncommittedTx.commit(lastCommittedTx);
        return firstUncommittedTx == entity ? entity : null;
    }
}
