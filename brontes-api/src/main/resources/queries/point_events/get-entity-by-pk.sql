SELECT
    branch_number,
    account_number,
    balance
FROM
    point_events
WHERE
    branch_number = :branchNumber
    AND account_number = :accountNumber