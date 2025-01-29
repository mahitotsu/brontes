package com.mahitotsu.brontes.api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.SpringBootTest.WebEnvironment;
import org.springframework.test.web.reactive.server.WebTestClient;

@SpringBootTest(webEnvironment = WebEnvironment.RANDOM_PORT)
public class HealthCheckTest {

    @Autowired
    private WebTestClient webClient;

    @Test
    public void testHealthCheck() {
        this.webClient.get().uri("/actuator/health").exchange().expectStatus().isOk();
    }
}
