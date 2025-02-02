package com.mahitotsu.brontes.api.repository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.r2dbc.core.R2dbcEntityOperations;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import com.mahitotsu.brontes.api.entity.PointEvent;

import reactor.core.publisher.Mono;

@Repository
public class PointEventRepository {

    @Autowired
    private R2dbcEntityOperations client;

    @Transactional
    public Mono<PointEvent> publishEvent(final String branchNumber, final String accountNumber, final int amount) {

        final PointEvent event = new PointEvent();
        event.setBranchNumber(branchNumber);
        event.setAccountNumber(accountNumber);
        event.setAmount(amount);

        return client.insert(event);
    }
}
