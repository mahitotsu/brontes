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
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Table(name = "account_transactions")
@Data
@Setter(AccessLevel.PRIVATE)
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class AccountTx {

    public static AccountTx newEntity(final Integer branchNumber, final Integer accountNumber,
            final BigDecimal amount) {

        final AccountTx entity = new AccountTx();
        entity.setBranchNumber(branchNumber);
        entity.setAccountNumber(accountNumber);
        entity.setAmount(amount);

        return entity;
    }

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

    @Column(name = "tx_status", insertable = false, updatable = true)
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

    public void commit(final AccountTx previousTx) {

        if (previousTx == null) {
            if (this.amount.compareTo(BigDecimal.ZERO) < 0) {
                this.txStatus = TxStatus.REJECTED;
                this.txSequence = 1;
                this.newBalance = BigDecimal.ZERO;
            } else {
                this.txStatus = TxStatus.ACCEPTED;
                this.txSequence = 1;
                this.newBalance = new BigDecimal(this.amount.toString());
            }
        } else if (previousTx.getTxStatus() == TxStatus.REGISTERED) {
            throw new IllegalStateException("The specified previous transaction status is not committed.");
        } else {
            final BigDecimal newBalance = previousTx.getNewBalance().add(this.amount);
            if (newBalance.compareTo(BigDecimal.ZERO) < 0 || newBalance.precision() - newBalance.scale() > 13) {
                this.txStatus = TxStatus.REJECTED;
                this.txSequence = previousTx.getTxSequence() + 1;
                this.newBalance = new BigDecimal(previousTx.getNewBalance().toString());
            } else {
                this.txStatus = TxStatus.ACCEPTED;
                this.txSequence = previousTx.getTxSequence() + 1;
                this.newBalance = newBalance;
            }
        }
    }
}
