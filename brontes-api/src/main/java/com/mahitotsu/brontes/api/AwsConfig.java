package com.mahitotsu.brontes.api;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.springframework.boot.autoconfigure.r2dbc.R2dbcProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import io.r2dbc.postgresql.PostgresqlConnectionConfiguration;
import io.r2dbc.postgresql.PostgresqlConnectionFactory;
import io.r2dbc.spi.ConnectionFactory;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.regions.providers.AwsRegionProvider;
import software.amazon.awssdk.regions.providers.DefaultAwsRegionProviderChain;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class AwsConfig {

    @Bean
    public AwsCredentialsProvider awsCredentialsProvider() {
        return DefaultCredentialsProvider.create();
    }

    @Bean
    public AwsRegionProvider awsRegionProvider() {
        return DefaultAwsRegionProviderChain.builder().build();
    }

    @Bean
    public ConnectionFactory connectionFactory(
            final R2dbcProperties properties,
            final AwsCredentialsProvider awsCredentialsProvider,
            final AwsRegionProvider awsRegionProvider) {

        final String[] segments = properties.getUrl().split(":");
        final String host = segments[2].split("/")[2];
        final String database = segments[2].split("/")[3].split("\\?")[0];
        final String username = properties.getUsername();

        final Region region = awsRegionProvider.getRegion();

        final Map<String, String> options = Arrays.stream(segments[2].split("/")[3].split("\\?")[1].split("&"))
                .collect(Collectors.toMap(item -> item.split("=")[0], item -> item.split("=")[1]));
        final DsqlUtilities utilities = DsqlUtilities
                .builder()
                .credentialsProvider(awsCredentialsProvider)
                .region(region)
                .build();
        final Supplier<CharSequence> token = () -> {
            return utilities.generateDbConnectAdminAuthToken(builder -> builder.hostname(host).region(region).build());
        };

        final PostgresqlConnectionConfiguration conf = PostgresqlConnectionConfiguration
                .builder()
                .host(host).database(database)
                .username(username).password(token).options(options)
                .build();
        return new PostgresqlConnectionFactory(conf);
    }
}
