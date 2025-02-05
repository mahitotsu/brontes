package com.mahitotsu.brontes.api.repository;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.r2dbc.core.DatabaseClient;

import com.mahitotsu.brontes.api.AbstractSpringTest;

import io.r2dbc.spi.IsolationLevel;
import reactor.core.publisher.Mono;

public class DatabaseConnectionTest extends AbstractSpringTest {

    @Autowired
    private DatabaseClient dbClient;

    @Test
    public void testConnectionConfiguration() {

        this.dbClient.inConnection((connection) -> {
            assertEquals(IsolationLevel.REPEATABLE_READ, connection.getTransactionIsolationLevel());
            assertFalse(connection.isAutoCommit());
            return Mono.from(connection.close());
        });
    }
}
