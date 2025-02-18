package com.mahitotsu.brontes.api.service.impl;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import java.util.stream.Stream;

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

    protected IdempotencyKey newIdempotencyKey(final UUID idemKey, final Map<String, Object> payload) {
        final IdempotencyKey entity = new IdempotencyKey();
        entity.setIdempotencyKey(idemKey);
        entity.setPayloadHash(this.calculatePayloadHash(payload));
        return entity;
    }

    protected <T> Mono<T> executeWithIdempotency(final UUID idemKey, final Map<String, Object> payload,
            final Callable<T> action) {
        return this.transactionalOperator.transactional(this.idempotencyKeyRepository.findById(idemKey)
                .flatMap(entity -> entity.getPayloadHash() == this.calculatePayloadHash(payload) ? Mono.just(entity)
                        : Mono.error(new RuntimeException()))
                .switchIfEmpty(this.idempotencyKeyRepository.save(this.newIdempotencyKey(idemKey, payload)))
                .then(Mono.fromCallable(action)))
                .retryWhen(RetrySpec.backoff(5, Duration.ofMillis(300)).jitter(0.5));
    }

    protected <T> Flux<T> executeWithIdempotency(final UUID idemKey, final Map<String, Object> payload,
            final Supplier<Stream<? extends T>> action) {
        return this.transactionalOperator.transactional(this.idempotencyKeyRepository.findById(idemKey)
                .switchIfEmpty(this.idempotencyKeyRepository.save(this.newIdempotencyKey(idemKey, payload)))
                .thenMany(Flux.fromStream(action)))
                .retryWhen(RetrySpec.backoff(5, Duration.ofMillis(300)).jitter(0.5));
    }
}
