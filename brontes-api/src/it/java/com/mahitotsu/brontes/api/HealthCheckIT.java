package com.mahitotsu.brontes.api;

import java.util.Properties;

import org.junit.jupiter.api.Test;

public class HealthCheckIT {

    @Test
    public void test() {
        Properties properties = System.getProperties();
        properties.list(System.out); 
    }
}
