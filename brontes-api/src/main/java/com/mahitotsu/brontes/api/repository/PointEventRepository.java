package com.mahitotsu.brontes.api.repository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.r2dbc.core.DatabaseClient;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import com.mahitotsu.brontes.api.entity.PointEvent;

import reactor.core.publisher.Mono;

@Repository
public class PointEventRepository {

    private static final String REPO_NAME = "point_events";

    @Autowired
    private DatabaseClient dbClient;

    @Autowired
    private QueryLoader queryExecutor;

    @Transactional
    public Mono<PointEvent> publishEvent(final String branchNumber, final String accountNumber, final int amount) {

        final int txSeq = 0;
        final Mono<String> insertResult = this.dbClient
                .sql(() -> this.queryExecutor.loadNamedQuery(REPO_NAME, "insert-new"))
                .bind("txSeq", txSeq)
                .bind("eventStatus", PointEvent.Status.C.name())
                .bind("branchNumber", branchNumber)
                .bind("accountNumber", accountNumber)
                .bind("amount", amount)
                .filter(stm -> stm.returnGeneratedValues("tx_id"))
                .map(row -> row.get("tx_id", String.class))
                .one();
        return insertResult.flatMap(txId -> this.dbClient
                .sql(() -> this.queryExecutor.loadNamedQuery(REPO_NAME, "select-by-pk"))
                .bind("txId", txId)
                .bind("txSeq", txSeq)
                .mapProperties(PointEvent.class)
                .one());
    }
}
