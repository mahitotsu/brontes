SELECT tx_seq
FROM point_events
WHERE branch_number = :branchNumber
    AND account_number = :accountNumber
ORDER BY tx_seq DESC
LIMIT 1