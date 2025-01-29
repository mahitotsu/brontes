package com.mahitotsu.brontes.api;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ResourceLoader;
import org.springframework.r2dbc.connection.init.ConnectionFactoryInitializer;
import org.springframework.r2dbc.connection.init.DatabasePopulator;
import org.springframework.r2dbc.connection.init.ResourceDatabasePopulator;

import io.r2dbc.spi.ConnectionFactory;

@TestConfiguration
public class TestEnvInitializer {

    @Autowired
    private ResourceLoader resourceLoader;

    @Bean
    public ConnectionFactoryInitializer connectionFactoryInitializer(final ConnectionFactory connectionFactory) {

        final DatabasePopulator creator = new ResourceDatabasePopulator(
                this.resourceLoader.getResource("initdb/schema-drop.sql"),
                this.resourceLoader.getResource("initdb/schema-create.sql"));
        final DatabasePopulator cleaner = new ResourceDatabasePopulator(
                this.resourceLoader.getResource("initdb/schema-drop.sql"));

        final ConnectionFactoryInitializer initializer = new ConnectionFactoryInitializer();
        initializer.setConnectionFactory(connectionFactory);
        initializer.setDatabasePopulator(creator);
        initializer.setDatabaseCleaner(cleaner);

        return initializer;
    }
}
