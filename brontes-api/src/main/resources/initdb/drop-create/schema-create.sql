CREATE TABLE IF NOT EXISTS accounts (
    branch_number CHAR(3) NOT NULL CHECK (branch_number ~ '^\d{3}$'),
    account_number CHAR(7) NOT NULL CHECK (account_number ~ '^\d{7}$'),
    balance BIGINT NOT NULL CHECK (balance >= 0),
    PRIMARY KEY (branch_number, account_number)
);