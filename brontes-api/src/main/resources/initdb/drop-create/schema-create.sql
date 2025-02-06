CREATE TABLE IF NOT EXISTS accounts (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    branch_number CHAR(3) NOT NULL,
    account_number CHAR(7) NOT NULL,
    balance BIGINT NOT NULL CHECK (balance >= 0),
    PRIMARY KEY (branch_number, account_number)
);