package com.mahitotsu.brontes.api.service.impl;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.function.Supplier;

import org.apache.commons.codec.digest.XXHash32;
import org.springframework.transaction.reactive.TransactionalOperator;

import com.mahitotsu.brontes.api.model.IdempotencyKey;
import com.mahitotsu.brontes.api.repository.IdempotencyKeyRepository;

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

    private IdempotencyKeyRepository idempotencyKeyRepository;

    private TransactionalOperator transactionalOperator;

    protected long calculatePayloadHash(final Map<String, Object> payload) {
        final XXHash32 encoder = new XXHash32();
        encoder.update(payload.toString().getBytes());
        return encoder.getValue();
    }

    protected MapBuilder newMapBuilder() {
        return new MapBuilder();
    }

    protected IdempotencyKey newIdempotencyKey(final UUID idemKey, final long payloadHash) {
        final IdempotencyKey entity = new IdempotencyKey();
        entity.setIdempotencyKey(idemKey);
        entity.setPayloadHash(payloadHash);
        return entity;
    }

    protected <T> Mono<? extends T> executeWithIdempotencyMono(final UUID idemKey, final Map<String, Object> payload,
            final Supplier<Mono<T>> action) {
        final long payloadHash = this.calculatePayloadHash(payload);
        return this.transactionalOperator.transactional(this.idempotencyKeyRepository.findById(idemKey)
                .flatMap(entity -> entity.getPayloadHash() == payloadHash ? Mono.just(entity) : Mono.error(new RuntimeException()))
                .switchIfEmpty(this.idempotencyKeyRepository.save(this.newIdempotencyKey(idemKey, payloadHash)))
                .then(action.get())
                .retryWhen(RetrySpec.backoff(5, Duration.ofMillis(300)).jitter(0.5)));
    }

    protected <T> Flux<? extends T> executeWithIdempotencyFlux(final UUID idemKey, final Map<String, Object> payload,
            final Supplier<Flux<T>> action) {
        final long payloadHash = this.calculatePayloadHash(payload);
        return this.transactionalOperator.transactional(this.idempotencyKeyRepository.findById(idemKey)
                .flatMap(entity -> entity.getPayloadHash() == payloadHash ? Mono.just(entity) : Mono.error(new RuntimeException()))
                .switchIfEmpty(this.idempotencyKeyRepository.save(this.newIdempotencyKey(idemKey, payloadHash)))
                .thenMany(action.get())
                .retryWhen(RetrySpec.backoff(5, Duration.ofMillis(300)).jitter(0.5)));
    }
}
