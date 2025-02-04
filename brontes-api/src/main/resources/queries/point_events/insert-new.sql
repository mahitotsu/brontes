INSERT INTO point_events (
        tx_seq,
        event_status,
        branch_number,
        account_number,
        amount
    )
VALUES (
        :txSeq,
        :eventStatus,
        :branchNumber,
        :accountNumber,
        :amount
    )