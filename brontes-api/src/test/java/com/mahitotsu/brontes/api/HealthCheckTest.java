package com.mahitotsu.brontes.api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.web.reactive.server.WebTestClient;

public class HealthCheckTest extends AbstractTest {

    @Autowired
    private WebTestClient webClient;

    @Test
    public void testHealthCheck() {
        this.webClient.get().uri("/actuator/health").exchange().expectStatus().isOk();
    }
}
