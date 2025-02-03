package com.mahitotsu.brontes.api.entity;

import java.util.UUID;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

import lombok.Data;

@Table("point_events")
@Data
public class PointEvent {

    public static enum Status {
        C, A, R
    }

    @Id
    private UUID eventId;
    private String transactionId;
    private Status eventStatus;
    private String branchNumber;
    private String accountNumber;
    private Integer amount;
}
