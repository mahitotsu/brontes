CREATE TABLE IF NOT EXISTS accounts (
    branch_number CHAR(3) NOT NULL CHECK (branch_number ~ '^\d{3}$'),
    account_number CHAR(7) NOT NULL CHECK (account_number ~ '^\d{7}$'),
    balance BIGINT NOT NULL CHECK (balance >= 0),
    PRIMARY KEY (branch_number, account_number)
);

CREATE TABLE IF NOT EXISTS transactions (
    tx_id UUID NOT NULL,
    tx_timestamp TIMESTAMP NOT NULL DEFAULT transaction_timestamp(),
    tx_type CHAR(1) NOT NULL CHECK (tx_type IN ('O', 'C', 'D', 'W')),
    branch_number CHAR(3) NOT NULL CHECK (branch_number ~ '^\d{3}$'),
    account_number CHAR(7) NOT NULL CHECK (account_number ~ '^\d{7}$'),
    amount BIGINT NOT NULL CHECK (amount >= 0),
    PRIMARY KEY (tx_id)
);