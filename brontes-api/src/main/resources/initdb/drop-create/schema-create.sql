DROP SCHEMA IF EXISTS idempotency_keys;
CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key UUID NOT NULL,
    payload_hash BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_payload_hash_length CHECK (octet_length(payload_hash) = 4),
    PRIMARY KEY (idempotency_key)
);

DROP TABLE IF EXISTS account_transactions;
CREATE TABLE IF NOT EXISTS account_transactions (
    tx_id UUID NOT NULL DEFAULT gen_random_uuid(),
    tx_seq BIGINT NOT NULL CHECK (tx_seq >= 0),
    tx_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    tx_type CHAR(1) NOT NULL CHECK (tx_type IN ('O', 'C', 'D', 'W')),
    branch_number CHAR(3) NOT NULL CHECK (branch_number ~ '^[0-9]{3}$'),
    account_number CHAR(7) NOT NULL CHECK (account_number ~ '^[0-9]{7}$'),
    amount NUMERIC(15,2) NOT NULL CHECK (amount >= 0),
    new_balance NUMERIC(15,2) NOT NULL CHECK (new_balance >= 0),
    UNIQUE (tx_seq, branch_number, account_number),
    PRIMARY KEY (tx_id)
);