package com.mahitotsu.brontes.api;

import java.sql.Connection;
import java.sql.SQLException;

import javax.sql.DataSource;

import org.postgresql.ds.PGSimpleDataSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.jdbc.datasource.DelegatingDataSource;
import org.springframework.lang.NonNull;

import com.zaxxer.hikari.HikariDataSource;

import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@SpringBootApplication
public class Main {

    public static void main(final String... args) {
        new SpringApplicationBuilder(Main.class).run(args);
    }

    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSourceProperties dataSourceProperties() {
        return new DataSourceProperties();
    }

    @Bean
    @Qualifier("targetDataSource")
    public DataSource targetDataSource(@Autowired final AwsCredentialsProvider awsCredentialsProvider) throws SQLException {

        final DataSourceProperties dataSourceProperties = dataSourceProperties();
        final PGSimpleDataSource targetDataSource = new PGSimpleDataSource();
        targetDataSource.setURL(dataSourceProperties.getUrl());

        final String endpoint = dataSourceProperties.getUrl().split("/")[2];
        final String username = dataSourceProperties.getUsername();
        final Region region = Region.of(endpoint.split("\\.")[2]);
        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(region)
                .credentialsProvider(awsCredentialsProvider).build();

        return new DelegatingDataSource(targetDataSource) {

            @Override
            public @NonNull Connection getConnection() throws SQLException {
                return this.getConnection(username, "dummy");
            }

            @Override
            public @NonNull Connection getConnection(final String username, final String password) throws SQLException {
                final String token = "admin".equals(username)
                        ? dsqlUtilities
                                .generateDbConnectAdminAuthToken(request -> request.hostname(endpoint).region(region))
                        : dsqlUtilities
                                .generateDbConnectAuthToken(request -> request.hostname(endpoint).region(region));
                return targetDataSource.getConnection(username, token);
            }
        };
    }

    @Bean
    @Primary
    public DataSource dataSource(@Qualifier("targetDataSource") final DataSource targetDataSource) {

        final DataSourceProperties dataSourceProperties = dataSourceProperties();
        final Class<? extends DataSource> dataSourceType = dataSourceProperties.getType();
        if (dataSourceType != null && HikariDataSource.class.isAssignableFrom(dataSourceType)) {
            throw new UnsupportedOperationException();
        }

        final HikariDataSource hikariDataSource = dataSourceProperties.initializeDataSourceBuilder()
                .type(HikariDataSource.class)
                .build();
        hikariDataSource.setDataSource(targetDataSource);
        return hikariDataSource;
    }
}
