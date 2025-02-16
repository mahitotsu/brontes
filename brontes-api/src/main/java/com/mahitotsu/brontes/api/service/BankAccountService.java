package com.mahitotsu.brontes.api.service;

import java.math.BigDecimal;
import java.util.UUID;

public interface BankAccountService {

    boolean openAccount(UUID idemKey, String branchNumber, String accountNumber);

    boolean closeAccount(UUID idemKey, String branchNumber, String accountNumber);

    BigDecimal deposit(UUID idemKey, String branchNumber, String accountNumber, BigDecimal amount);

    BigDecimal withdraw(UUID idemKey, String branchNumber, String accountNumber, BigDecimal amount);
}
