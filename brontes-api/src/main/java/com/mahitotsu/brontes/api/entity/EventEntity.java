package com.mahitotsu.brontes.api.entity;

import java.time.LocalDateTime;
import java.util.UUID;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

import lombok.Data;

@Data
@Table("events")
public class EventEntity {

    public static enum EventType {
        A, S,
    }

    public static enum EventStatus {
        C, A, R,
    }

    @Id
    private UUID eventId;
    private EventType eventType;
    private EventStatus eventStatus;
    private LocalDateTime eventTimestamp;
    private String branchNumber;
    private String accountNumber;
    private long amount;
}
