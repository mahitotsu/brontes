INSERT INTO point_events (
        branch_number,
        account_number,
        tx_seq,
        amount
    )
VALUES (
        :branchNumber,
        :accountNumber,
        :newTxSeq,
        :amount
    )