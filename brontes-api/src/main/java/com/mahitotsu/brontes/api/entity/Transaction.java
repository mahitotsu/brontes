package com.mahitotsu.brontes.api.entity;

import java.time.LocalDateTime;
import java.util.UUID;

import org.springframework.data.relational.core.mapping.Table;

import lombok.Data;

@Table("transactions")
@Data
public class Transaction {
   private UUID txId;
   private LocalDateTime txTimestamp;
   private TransactionType txType;
   private String branchNumber;
   private String accountNumber; 
   private Long amount;
}
