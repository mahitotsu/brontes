package com.mahitotsu.brontes.api.service.impl;

import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import org.apache.commons.codec.digest.XXHash32;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.data.r2dbc.core.R2dbcEntityOperations;
import org.springframework.transaction.ReactiveTransactionManager;
import org.springframework.transaction.reactive.TransactionalOperator;
import org.springframework.transaction.support.DefaultTransactionDefinition;

import com.mahitotsu.brontes.api.model.IdempotencyKey;
import com.mahitotsu.brontes.api.repository.IdempotencyKeyRepository;

import jakarta.annotation.PostConstruct;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.util.retry.RetrySpec;

public abstract class AbstractIdempotentService {

    protected static class MapBuilder {

        private final Map<String, Object> map;

        public MapBuilder() {
            this.map = new HashMap<>();
        }

        public MapBuilder put(final String key, final Object value) {
            map.put(key, value);
            return this;
        }

        public Map<String, Object> build() {
            return map;
        }
    }

    @Autowired
    private ReactiveTransactionManager transactionManager;

    private TransactionalOperator rwTxOperator;

    private TransactionalOperator roTxOperator;

    @Autowired
    private IdempotencyKeyRepository idempotencyKeyRepository;

    @Autowired
    private R2dbcEntityOperations entityOperations;


    @PostConstruct
    public void init() {

        final DefaultTransactionDefinition rwTxDef = new DefaultTransactionDefinition();
        rwTxDef.setReadOnly(false);
        final DefaultTransactionDefinition roTxDef = new DefaultTransactionDefinition();
        roTxDef.setReadOnly(true);

        this.rwTxOperator = TransactionalOperator.create(this.transactionManager, rwTxDef);
        this.roTxOperator = TransactionalOperator.create(this.transactionManager, roTxDef);
    }

    protected byte[] calculatePayloadHash(final Map<String, Object> payload) {
        final XXHash32 encoder = new XXHash32();
        encoder.update(payload.toString().getBytes());

        final ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES);
        buffer.putInt(payload.hashCode());
        return buffer.array();
    }

    protected MapBuilder newMapBuilder() {
        return new MapBuilder();
    }

    protected IdempotencyKey newIdempotencyKey(final UUID idemKey, final byte[] payloadHash) {
        final IdempotencyKey entity = new IdempotencyKey();
        entity.setIdempotencyKey(idemKey);
        entity.setPayloadHash(payloadHash);
        return entity;
    }

    protected <T> Mono<T> executeWithIdempotencyMono(final UUID idemKey, final Map<String, Object> payload,
            final Mono<T> action) {
        final byte[] payloadHash = this.calculatePayloadHash(payload);
        return this.rwTxOperator.transactional(this.idempotencyKeyRepository.findById(idemKey)
                .flatMap(entity -> Arrays.equals(entity.getPayloadHash(), payloadHash) ? Mono.just(entity)
                        : Mono.error(new RuntimeException()))
                .switchIfEmpty(this.entityOperations.insert(this.newIdempotencyKey(idemKey, payloadHash)))
                .then(action)
                .retryWhen(RetrySpec.backoff(5, Duration.ofMillis(300)).jitter(0.5)
                        .filter(t -> CannotAcquireLockException.class.isInstance(t))));
    }

    protected <T> Flux<T> executeWithIdempotencyFlux(final UUID idemKey, final Map<String, Object> payload,
            final Flux<T> action) {
        final byte[] payloadHash = this.calculatePayloadHash(payload);
        return this.rwTxOperator.transactional(this.idempotencyKeyRepository.findById(idemKey)
                .flatMap(entity -> Arrays.equals(entity.getPayloadHash(), payloadHash) ? Mono.just(entity)
                        : Mono.error(new RuntimeException()))
                .switchIfEmpty(this.entityOperations.insert(this.newIdempotencyKey(idemKey, payloadHash)))
                .thenMany(action)
                .retryWhen(RetrySpec.backoff(5, Duration.ofMillis(300)).jitter(0.5)
                        .filter(t -> CannotAcquireLockException.class.isInstance(t))));
    }

    protected <T> Mono<T> queryMono(final Mono<T> action) {
        return this.roTxOperator.transactional(action);
    }

    protected <T> Flux<T> queryFlux(final Flux<T> action) {
        return this.roTxOperator.transactional(action);
    }
}
