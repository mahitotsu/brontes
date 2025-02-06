UPDATE
    point_events
SET
    (balance = :balance)
WHERE
    branch_number = :branchNumber
    AND account_number = :accountNumber