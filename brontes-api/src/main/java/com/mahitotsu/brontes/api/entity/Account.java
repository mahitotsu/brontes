package com.mahitotsu.brontes.api.entity;

import org.springframework.data.relational.core.mapping.Table;

import lombok.Data;

@Table("accounts")
@Data
public class Account {
    private String branchNumber;
    private String accountNumber;
    private Long balance;
}
