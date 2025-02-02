CREATE TABLE IF NOT EXISTS point_events (
    event_id UUID NOT NULL DEFAULT gen_random_uuid(),
    transaction_id CHAR(20) NOT NULL DEFAULT txid_current()::text,
    event_status CHAR(1) NOT NULL DEFAULT 'C', -- (C)reate/(A)ccepted/(R)ejected
    branch_number CHAR(3) NOT NULL,
    account_number CHAR(7) NOT NULL,
    amount INTEGER NOT NULL,
    PRIMARY KEY (event_id)
);
