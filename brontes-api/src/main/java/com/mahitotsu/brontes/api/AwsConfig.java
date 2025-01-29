package com.mahitotsu.brontes.api;

import java.util.function.Supplier;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.r2dbc.ConnectionFactoryOptionsBuilderCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import io.r2dbc.spi.ConnectionFactoryOptions;
import io.r2dbc.spi.Option;
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
    public ConnectionFactoryOptionsBuilderCustomizer connectionFactoryOptionsBuilderCustomizer(
            final AwsCredentialsProvider awsCredentialsProvider,
            @Value("${AWS_DSQL_ENDPOINT}") final String dsqlEndpoint) {

        final Region region = Region.of(dsqlEndpoint.split("\\.")[2]);
        final DsqlUtilities utilities = DsqlUtilities
                .builder()
                .credentialsProvider(awsCredentialsProvider)
                .region(region)
                .build();
        final Supplier<CharSequence> token = () -> {
            return utilities.generateDbConnectAdminAuthToken(builder -> builder.hostname(dsqlEndpoint).build());
        };

        return builder -> {
            builder.option(Option.sensitiveValueOf(ConnectionFactoryOptions.PASSWORD.name()), token);
        };
    }
}
