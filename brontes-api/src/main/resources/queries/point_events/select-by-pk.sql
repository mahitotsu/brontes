SELECT CONCAT(tx_id, lpad(to_hex(tx_seq), 8, '0')) AS event_id,
    event_status,
    branch_number,
    account_number,
    amount
FROM point_events
WHERE tx_id = :txId
    AND tx_seq = :txSeq