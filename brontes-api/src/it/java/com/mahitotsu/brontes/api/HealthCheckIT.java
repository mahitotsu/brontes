package com.mahitotsu.brontes.api;

import org.junit.jupiter.api.Test;

import io.restassured.RestAssured;

public class HealthCheckIT extends RestApiTest {

    @Test
    public void testHealth() {
        RestAssured.given().when().get("/actuator/health").then().statusCode(200);
    }
}
