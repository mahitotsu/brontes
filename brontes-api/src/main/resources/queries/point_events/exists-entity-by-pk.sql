SELECT
    count(account_number) > 0
FROM
    point_events
WHERE
    branch_number = :branchNumber
    AND account_number = :accountNumber