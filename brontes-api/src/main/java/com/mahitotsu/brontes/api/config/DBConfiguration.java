package com.mahitotsu.brontes.api.config;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.time.Duration;
import java.util.Arrays;

import javax.sql.DataSource;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.datasource.AbstractDataSource;

import com.zaxxer.hikari.HikariDataSource;

import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class DBConfiguration {

    @Value("${DSQL_ENDPOINTS}")
    private String[] dsqlEndpoints;

    @Value("${AWS_REGION}")
    private String region;

    @Bean
    public DataSource dataSource(final AwsCredentialsProvider awsCredentialsProvider) {

        final Region region = Region.of(this.region);
        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(region)
                .credentialsProvider(awsCredentialsProvider)
                .build();

        final String dsqlEndpoint = Arrays.stream(this.dsqlEndpoints).filter(endpoint -> endpoint.contains(this.region))
                .findFirst().get();
        final String url = "jdbc:postgresql://" + dsqlEndpoint + "/postgres?sslmode=require";

        final DataSource dsqlDS = new AbstractDataSource() {

            @Override
            public Connection getConnection() throws SQLException {
                final String username = "admin";
                final String token = dsqlUtilities.generateDbConnectAdminAuthToken(
                        builder -> builder.hostname(dsqlEndpoint)
                                .region(region)
                                .expiresIn(Duration.ofSeconds(10))
                                .build());
                return this.getConnection(username, token);
            }

            @Override
            public Connection getConnection(final String username, final String password) throws SQLException {
                return DriverManager.getConnection(url, username, password);
            }
        };

        final DataSourceProperties props = new DataSourceProperties();
        props.setUrl(url);

        final HikariDataSource hikariCp = props.initializeDataSourceBuilder()
                .type(HikariDataSource.class).build();
        hikariCp.setAutoCommit(false);
        hikariCp.setDataSource(dsqlDS);
        hikariCp.setIdleTimeout(0);
        hikariCp.setMaximumPoolSize(1);

        return hikariCp;
    }

}
