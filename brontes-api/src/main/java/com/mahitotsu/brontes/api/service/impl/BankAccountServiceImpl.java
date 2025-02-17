package com.mahitotsu.brontes.api.service.impl;

import java.math.BigDecimal;
import java.util.UUID;

import org.springframework.stereotype.Service;

import com.mahitotsu.brontes.api.service.BankAccountService;

@Service
public class BankAccountServiceImpl extends AbstractIdempotentService implements BankAccountService {

    @Override
    public boolean openAccount(final UUID idemKey, final String branchNumber, final String accountNumber) {
        return false;
    }

    @Override
    public boolean closeAccount(final UUID idemKey, final String branchNumber, final String accountNumber) {
        return false;
    }

    @Override
    public BigDecimal deposit(final UUID idemKey, final String branchNumber, final String accountNumber,
            final BigDecimal amount) {
        return null;
    }

    @Override
    public BigDecimal withdraw(final UUID idemKey, final String branchNumber, final String accountNumber,
            final BigDecimal amount) {
        return null;
    }
}
