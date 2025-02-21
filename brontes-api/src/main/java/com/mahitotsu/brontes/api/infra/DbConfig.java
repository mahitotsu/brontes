package com.mahitotsu.brontes.api.infra;

import java.sql.Connection;
import java.sql.SQLException;

import javax.sql.DataSource;

import org.postgresql.ds.PGSimpleDataSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.jdbc.datasource.DelegatingDataSource;
import org.springframework.lang.NonNull;

import com.zaxxer.hikari.HikariDataSource;
import com.zaxxer.hikari.SQLExceptionOverride;

import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class DbConfig {

    static public class DsqlExceptionOverride implements SQLExceptionOverride {

        @java.lang.Override
        public Override adjudicate(final SQLException e) {
            final String sqlState = e.getSQLState();
            if ("0C000".equals(sqlState) || "0C001".equals(sqlState) || sqlState.matches("0A\\d{3}")) {
                return Override.DO_NOT_EVICT;
            }
            return Override.CONTINUE_EVICT;
        }
    };

    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSourceProperties dataSourceProperties() {
        return new DataSourceProperties();
    }

    @Bean
    @Qualifier("targetDataSource")
    public DataSource targetDataSource(@Autowired final AwsCredentialsProvider awsCredentialsProvider)
            throws SQLException {

        final DataSourceProperties dataSourceProperties = dataSourceProperties();
        final PGSimpleDataSource targetDataSource = new PGSimpleDataSource();
        targetDataSource.setURL(dataSourceProperties.getUrl());

        final String endpoint = dataSourceProperties.getUrl().split("/")[2];
        final String username = dataSourceProperties.getUsername();
        final String password = dataSourceProperties.getPassword();
        final Region region = Region.of(endpoint.split("\\.")[2]);
        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(region)
                .credentialsProvider(awsCredentialsProvider).build();

        return new DelegatingDataSource(targetDataSource) {

            @Override
            public @NonNull Connection getConnection() throws SQLException {
                return this.getConnection(username, password);
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
        hikariDataSource.setExceptionOverrideClassName(DsqlExceptionOverride.class.getName());
        return hikariDataSource;
    }
}
