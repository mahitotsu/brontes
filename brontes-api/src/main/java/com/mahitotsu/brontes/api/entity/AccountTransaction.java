package com.mahitotsu.brontes.api.entity;

import java.math.BigDecimal;
import java.time.ZonedDateTime;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Setter;

@Entity
@Table(name = "account_transactions")
@Data
@Setter(AccessLevel.PRIVATE)
public class AccountTransaction {

    public static enum TxStatus {
        REGISTERED, ACCEPTED, REJECTED,
    }

    @GeneratedValue
    @Id
    @Column(name = "tx_id", insertable = false, updatable = false)
    private UUID txId;

    @Column(name = "tx_sequence", insertable = false, updatable = true)
    private Integer txSequence;

    @Column(name = "tx_timestamp", insertable = false, updatable = false)
    private ZonedDateTime txTimestamp;

    @Column(name = "txStatus", insertable = false, updatable = true)
    @Enumerated(EnumType.ORDINAL)
    private TxStatus txStatus;

    @Column(name = "branch_number", insertable = true, updatable = false)
    private Integer branchNumber;

    @Column(name = "account_number", insertable = true, updatable = false)
    private Integer accountNumber;

    @Column(name = "amount", insertable = true, updatable = false)
    private BigDecimal amount;

    @Column(name = "new_balance", insertable = false, updatable = true)
    private BigDecimal newBalance;

    @SuppressWarnings("unused")
    private AccountTransaction() {
    }

    public AccountTransaction(final Integer branchNumber, final Integer accountNumber, final BigDecimal amount) {
        this.branchNumber = branchNumber;
        this.accountNumber = accountNumber;
        this.amount = amount;
    }

    public void accept(final AccountTransaction previousTx) {

        if (previousTx == null) {
            this.txStatus = TxStatus.ACCEPTED;
            this.txSequence = 1;
            this.newBalance = new BigDecimal(this.amount.toString());
        } else if (previousTx.getTxStatus() == TxStatus.REGISTERED) {
            throw new IllegalStateException("The specified previous transaction status is not committed.");
        } else {
            this.txStatus = TxStatus.ACCEPTED;
            this.txSequence = previousTx.getTxSequence() + 1;
            this.newBalance = previousTx.newBalance.add(this.amount);
        }
    }

    public void reject(final AccountTransaction previousTx) {

        if (previousTx == null) {
            this.txStatus = TxStatus.REJECTED;
            this.txSequence = 1;
            this.newBalance = BigDecimal.ZERO;
        } else if (previousTx.getTxStatus() == TxStatus.REGISTERED) {
            throw new IllegalStateException("The specified previous transaction status is not committed.");
        } else {
            this.txStatus = TxStatus.REJECTED;
            this.txSequence = previousTx.getTxSequence() + 1;
            this.newBalance = new BigDecimal(previousTx.getNewBalance().toString());
        }
    }
}
