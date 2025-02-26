package com.mahitotsu.brontes.api.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.mahitotsu.brontes.api.repository.AccountTransactionRepository;

@Service
public class AccountTransactionService {

    @Autowired
    private AccountTransactionRepository accountTransactionRepository;

    public void openAccount(final String branchCode, final String accountNumber) {

    }

    public void closeAccount(final String branchCode, final String accountNumber) {

    }

    public void deposit(final String branchCode, final String accountNumber, final int amount) {

    }

    public void withdraw(final String branchCode, final String accountNumber, final int amount) {

    }

    public int getBalance(final String branchCode, final String accountNumber) {
        return 0;
    }
}
