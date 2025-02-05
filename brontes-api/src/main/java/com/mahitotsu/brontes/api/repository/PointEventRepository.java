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

        return Mono.just(0L)
                .flatMap(initialTxSeq -> this.dbClient
                        .sql(() -> this.queryExecutor.loadNamedQuery(REPO_NAME, "get-last-txSeq"))
                        .bind("branchNumber", branchNumber)
                        .bind("accountNumber", accountNumber)
                        .mapValue(Long.class)
                        .one()
                        .defaultIfEmpty(initialTxSeq)
                        .map(lastTxSeq -> lastTxSeq + 1))
                .flatMap(newTxSeq -> this.dbClient
                        .sql(() -> this.queryExecutor.loadNamedQuery(REPO_NAME, "insert-new-entity"))
                        .bind("branchNumber", branchNumber)
                        .bind("accountNumber", accountNumber)
                        .bind("newTxSeq", newTxSeq)
                        .bind("amount", amount)
                        .fetch()
                        .rowsUpdated()
                        .thenReturn(newTxSeq))
                .flatMap(newTxSeq -> this.dbClient
                        .sql(() -> this.queryExecutor.loadNamedQuery(REPO_NAME, "get-entity-by-pk"))
                        .bind("branchNumber", branchNumber)
                        .bind("accountNumber", accountNumber)
                        .bind("newTxSeq", newTxSeq)
                        .mapProperties(PointEvent.class)
                        .one());
    }
}
