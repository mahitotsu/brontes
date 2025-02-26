DROP TABLE IF EXISTS account_transactions;

CREATE TABLE IF NOT EXISTS account_transactions (
    -- immutable columns, set by system
    tx_id UUID NOT NULL DEFAULT gen_random_uuid(),
    tx_timestamp TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    -- immutable columns, set by user
    branch_number SMALLINT NOT NULL CHECK (
        branch_number >= 0
        AND branch_number < 1000
    ),
    account_number INTEGER NOT NULL CHECK (
        account_number >= 0
        AND account_number < 10000000
    ),
    amount NUMERIC(15, 2) NOT NULL,
    -- updatable only after inserted by user
    tx_status SMALLINT NOT NULL DEFAULT 0 CHECK (tx_status IN (0, 1, 2)),
    tx_sequence INTEGER,
    new_balance NUMERIC(15, 2),
    -- constraints
    CHECK (
        (
            tx_status <> 0
            AND tx_sequence > 0
        )
        OR (
            tx_status = 0
            AND tx_sequence IS NULL
        )
    ),
    CHECK (
        (
            tx_status <> 0
            AND new_balance >= 0
        )
        OR (
            tx_status = 0
            AND new_balance IS NULL
        )
    ),
    UNIQUE (branch_number, account_number, tx_sequence),
    PRIMARY KEY (tx_id)
);