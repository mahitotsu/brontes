SELECT branch_number,
    account_number,
    tx_seq,
    amount
FROM point_events
WHERE branch_number = :branchNumber
    AND account_number = :accountNumber
    AND tx_seq = :newTxSeq