CREATE TABLE IF NOT EXISTS point_events (
    branch_number CHAR(3) NOT NULL,
    account_number CHAR(7) NOT NULL,
    tx_seq BIGINT NOT NULL,
    amount INTEGER NOT NULL,
    PRIMARY KEY (branch_number, account_number, tx_seq)
);
