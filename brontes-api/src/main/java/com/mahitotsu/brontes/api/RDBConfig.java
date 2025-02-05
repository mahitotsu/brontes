package com.mahitotsu.brontes.api;

import java.time.Duration;
import java.util.function.Supplier;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.r2dbc.ConnectionFactoryOptionsBuilderCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.ResourceLoader;
import org.springframework.r2dbc.connection.init.ConnectionFactoryInitializer;
import org.springframework.r2dbc.connection.init.DatabasePopulator;
import org.springframework.r2dbc.connection.init.ResourceDatabasePopulator;

import io.r2dbc.spi.ConnectionFactory;
import io.r2dbc.spi.ConnectionFactoryOptions;
import io.r2dbc.spi.Option;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class RDBConfig {

    @Autowired
    private ResourceLoader resourceLoader;

    @Bean
    public ConnectionFactoryOptionsBuilderCustomizer connectionFactoryOptionsBuilderCustomizer(
            final AwsCredentialsProvider awsCredentialsProvider,
            @Value("${spring.r2dbc.url}") final String url) {

        final String endpoint = url.split("/")[2];
        final Region region = Region.of(endpoint.split("\\.")[2]);
        final DsqlUtilities utilities = DsqlUtilities
                .builder()
                .credentialsProvider(awsCredentialsProvider)
                .region(region)
                .build();
        final Supplier<CharSequence> tokenSupplier = () -> utilities.generateDbConnectAdminAuthToken(builder -> builder
                .hostname(endpoint)
                .expiresIn(Duration.ofSeconds(10))
                .build());

        return builder -> {
            builder.option(Option.sensitiveValueOf(ConnectionFactoryOptions.PASSWORD.name()), tokenSupplier);
        };
    }

    @Bean
    @Profile("initdb")
    public ConnectionFactoryInitializer connectionFactoryInitializer(final ConnectionFactory connectionFactory) {

        final DatabasePopulator creator = new ResourceDatabasePopulator(
                this.resourceLoader.getResource("classpath:queries/initdb/schema-drop.sql"),
                this.resourceLoader.getResource("classpath:queries/initdb/schema-create.sql"));
        final DatabasePopulator cleaner = new ResourceDatabasePopulator(
                this.resourceLoader.getResource("classpath:queries/initdb/schema-drop.sql"));

        final ConnectionFactoryInitializer initializer = new ConnectionFactoryInitializer();
        initializer.setConnectionFactory(connectionFactory);
        initializer.setDatabasePopulator(creator);
        initializer.setDatabaseCleaner(cleaner);

        return initializer;
    }
}
