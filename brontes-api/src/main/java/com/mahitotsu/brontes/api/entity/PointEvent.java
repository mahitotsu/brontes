package com.mahitotsu.brontes.api.entity;

import lombok.Data;

@Data
public class PointEvent {

    public static enum Status {
        C, A, R
    }

    private String branchNumber;
    private String accountNumber;
    private Long txSeq;
    private Integer amount;
}
