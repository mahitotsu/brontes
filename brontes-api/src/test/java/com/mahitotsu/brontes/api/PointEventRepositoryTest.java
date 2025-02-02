package com.mahitotsu.brontes.api;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.annotation.DirtiesContext.ClassMode;

import com.mahitotsu.brontes.api.config.DataConfig;
import com.mahitotsu.brontes.api.entity.PointEvent;
import com.mahitotsu.brontes.api.repository.PointEventRepository;

import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

@SpringBootTest
@Import({ DataConfig.class })
@DirtiesContext(classMode = ClassMode.AFTER_CLASS)
public class PointEventRepositoryTest {

    private static final Random SEED = new Random();

    @Autowired
    private PointEventRepository pointEventRepository;

    @Test
    public void testPublishPointEvent() {

        final String branchNumber = String.format("%03d", SEED.nextInt(10 ^ 3));
        final String accountNumber = String.format("%07d", SEED.nextInt(10 ^ 7));
        final int amount = SEED.nextInt(SEED.nextInt(100) * (SEED.nextBoolean() ? 1 : -1));

        final Mono<PointEvent> mono = this.pointEventRepository.publishEvent(branchNumber, accountNumber, amount);

        StepVerifier.create(mono).expectNextMatches(entity -> {
            return branchNumber.equals(entity.getBranchNumber())
                    && accountNumber.equals(entity.getAccountNumber())
                    && amount == entity.getAmount();
        }).verifyComplete();
    }
}
