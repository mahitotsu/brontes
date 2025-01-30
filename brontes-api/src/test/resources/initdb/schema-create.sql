CREATE TABLE IF NOT EXISTS events (
    event_id UUID NOT NULL DEFAULT gen_random_uuid(),
    event_type char(1) NOT NULL, -- (A)dd, (S)ub
    event_status char(1) NOT NULL DEFAULT 'C', -- (C)reated, (A)ccepted, (R)jected
    event_timestamp TIMESTAMP(3) NOT NULL DEFAULT clock_timestamp(),
    branch_number char(3) NOT NULL,
    account_number char(7) NOT NULL,
    amount INTEGER NOT NULL CHECK (amount > 0),
    PRIMARY KEY (event_id)
);
