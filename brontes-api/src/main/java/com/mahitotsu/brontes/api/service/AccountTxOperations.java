package com.mahitotsu.brontes.api.service;

import java.math.BigDecimal;
import java.time.ZonedDateTime;
import java.util.UUID;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;

import com.mahitotsu.brontes.api.entity.AccountTx;
import com.mahitotsu.brontes.api.entity.AccountTx.TxStatus;
import com.mahitotsu.brontes.api.repository.AccountTxRepository;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

@Component
@Validated
public class AccountTxOperations {

    private AccountTxRepository accountTxRepository;

    @Transactional(readOnly = true)
    public boolean existsAccount(
            @NotNull @Pattern(regexp = "^[0-9]{3}$") final String branchNumber,
            @NotNull @Pattern(regexp = "^[0-9]{7}$") final String accountNumber) {

        return this.accountTxRepository.existsByBranchNumberAndAccountNumber(Integer.valueOf(branchNumber),
                Integer.valueOf(accountNumber));
    }

    @Transactional(readOnly = true)
    public Integer getLastBalance(
            @NotNull @Pattern(regexp = "^[0-9]{3}$") final String branchNumber,
            @NotNull @Pattern(regexp = "^[0-9]{7}$") final String accountNumber,
            final ZonedDateTime txTimestamp) {

        return this.accountTxRepository.findOneLastCommittedTransaction(Integer.valueOf(branchNumber),
                Integer.valueOf(accountNumber)).map(entity -> entity.getNewBalance().intValue()).orElse(null);
    }

    @Transactional
    @DsqlRetry
    public AccountTx registerAccountTx(
            @NotNull @Pattern(regexp = "^[0-9]{3}$") final String branchNumber,
            @NotNull @Pattern(regexp = "^[0-9]{7}$") final String accountNumber,
            final int amount) {

        return this.accountTxRepository
                .findById(this.accountTxRepository.save(AccountTx.newEntity(Integer.valueOf(branchNumber),
                        Integer.valueOf(accountNumber), new BigDecimal(amount))).getTxId())
                .orElseThrow(() -> new IllegalStateException("An unexpected situation has occurred."));
    }

    @Transactional
    @DsqlRetry
    public AccountTx commitAccountTx(@NotNull final UUID txId) {

        final AccountTx entity = this.accountTxRepository.findById(txId).get();
        if (entity.getTxStatus() != TxStatus.REGISTERED) {
            return entity;
        }

        final Integer branchNumber = entity.getBranchNumber();
        final Integer accountNumber = entity.getAccountNumber();
        AccountTx.commitTransactions(
                this.accountTxRepository.findOneLastCommittedTransaction(branchNumber, accountNumber).orElse(null),
                this.accountTxRepository.findAllUncommittedTransactions(branchNumber, accountNumber,
                        entity.getTxTimestamp()));

        return this.accountTxRepository.findById(txId).filter(atx -> atx.getTxStatus() != TxStatus.REGISTERED)
                .orElseThrow(() -> new IllegalStateException("An unexpected situation has occurred."));
    }

    @Transactional(readOnly = true)
    public Integer regirievePastBalance(@NotNull final UUID txId) {

        return this.accountTxRepository.findById(txId).filter(entity -> entity.getTxSequence() != null)
                .map((entity -> entity.getNewBalance().intValue())).orElse(null);
    }
}
