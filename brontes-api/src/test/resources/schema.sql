DROP TABLE IF EXISTS account_transactions;
CREATE TABLE IF NOT EXISTS acount_transactions (
    id UUID NOT NULL DEFAULT gen_random_uuid (),
    PRIMARY KEY (id)
);