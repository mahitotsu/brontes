package com.mahitotsu.brontes.api.service;

public class AccountTxRejectedException extends RuntimeException {
   
    public AccountTxRejectedException(final String message) {
        super(message);
    }
}
