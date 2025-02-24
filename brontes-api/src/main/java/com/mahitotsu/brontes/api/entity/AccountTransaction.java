package com.mahitotsu.brontes.api.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Data;

@Entity
@Table(name = "account_transactions")
@Data
public class AccountTransaction {

    @Id
    @Column(name = "id", insertable = false, updatable = false)
    private UUID id;
}
