package com.mahitotsu.brontes.api;

import org.junit.jupiter.api.BeforeAll;

import io.restassured.RestAssured;
import io.restassured.builder.RequestSpecBuilder;

public abstract class RestApiTest {

    @BeforeAll
    public static void setup() {
        RestAssured.baseURI = System.getenv("NGROK_URL");
        RestAssured.requestSpecification = new RequestSpecBuilder().addHeader("Content-Type", "application/json")
                .build();
    }
}
