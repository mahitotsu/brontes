CREATE TABLE IF NOT EXISTS point_events (
    tx_id CHAR(16) NOT NULL DEFAULT lpad(to_hex(pg_current_xact_id()::text::bigint), 16, '0'),
    tx_seq INTEGER NOT NULL,
    event_status CHAR(1) NOT NULL DEFAULT 'C', -- (C)reate/(A)ccepted/(R)ejected
    branch_number CHAR(3) NOT NULL,
    account_number CHAR(7) NOT NULL,
    amount INTEGER NOT NULL,
    PRIMARY KEY (tx_id, tx_seq)
);
