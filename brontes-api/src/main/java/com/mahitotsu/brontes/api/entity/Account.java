package com.mahitotsu.brontes.api.entity;

import java.util.UUID;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

import lombok.Data;

@Table("accounts")
@Data
public class Account {
    @Id
    private UUID id;
    private String branchNumber;
    private String accountNumber;
    private Long balance;
}
