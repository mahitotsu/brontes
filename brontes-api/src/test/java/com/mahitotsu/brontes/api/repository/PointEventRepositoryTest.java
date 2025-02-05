package com.mahitotsu.brontes.api.repository;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

import com.mahitotsu.brontes.api.AbstractSpringTest;
import com.mahitotsu.brontes.api.entity.PointEvent;

import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

public class PointEventRepositoryTest extends AbstractSpringTest {

    private static final Random SEED = new Random();

    @Autowired
    private PointEventRepository pointEventRepository;

    @Test
    public void testPublishPointEvent() {

        final String branchNumber = String.format("%03d", SEED.nextInt(100));
        final String accountNumber = String.format("%07d", SEED.nextInt(10000000));
        final int amount = SEED.nextInt(100) * (SEED.nextBoolean() ? 1 : -1);

        final Mono<PointEvent> mono = this.pointEventRepository.publishEvent(branchNumber, accountNumber, amount);

        StepVerifier.create(mono).assertNext(entity -> {
            assertEquals(branchNumber, entity.getBranchNumber());
            assertEquals(accountNumber, entity.getAccountNumber());
            assertEquals(amount, entity.getAmount());
            assertNotNull(entity.getTxSeq());
        }).verifyComplete();
    }

    @Test
    public void testGeneratesUniqueAndOrderedEvents() throws Exception {

        final String branchNumber = String.format("%03d", SEED.nextInt(100));
        final String accountNumber = String.format("%07d", SEED.nextInt(10000000));
        final Callable<Mono<PointEvent>> publisEventTask = () -> {
            final int amount = SEED.nextInt(100) * (SEED.nextBoolean() ? 1 : -1);
            return this.pointEventRepository.publishEvent(branchNumber, accountNumber, amount);
        };

        final List<PointEvent> eventList = new ArrayList<>();
        final Mono<PointEvent> publish1 = publisEventTask.call().doOnNext(eventList::add);
        final Mono<PointEvent> publish2 = publisEventTask.call().doOnNext(eventList::add);

        StepVerifier.create(publish1.then(publish2).then(Mono.just(eventList))).assertNext(list -> {
            assertEquals(2, list.size());
            final PointEvent event1 = list.get(0);
            final PointEvent event2 = list.get(1);
            assertNotEquals(event1.getTxSeq(), event2.getTxSeq());
            assertTrue(event1.getTxSeq().compareTo(event2.getTxSeq()) < 0);
        }).verifyComplete();
    }
}
