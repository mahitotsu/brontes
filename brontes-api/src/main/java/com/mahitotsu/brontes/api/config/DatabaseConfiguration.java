package com.mahitotsu.brontes.api.config;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.time.Duration;
import java.util.Properties;

import javax.sql.DataSource;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.datasource.AbstractDataSource;

import com.zaxxer.hikari.HikariDataSource;

import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class DatabaseConfiguration {

    @Value("${DSQL_ENDPOINT}")
    private String dsqlEndpoint;

    @Bean
    public DataSource dataSource(final AwsCredentialsProvider awsCredentialsProvider) {

        final String hostname = this.dsqlEndpoint;
        final Region region = Region.of(hostname.split("\\.")[2]);
        final String url = "jdbc:postgresql://" + hostname + "/postgres";

        final Properties props = new Properties();
        props.setProperty("ssl", "true");
        props.setProperty("sslmode", "require");

        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().credentialsProvider(awsCredentialsProvider)
                .region(region).build();
        final DataSource dataSource = new AbstractDataSource() {

            @Override
            public Connection getConnection() throws SQLException {

                final String username = "admin";
                final String password = dsqlUtilities.generateDbConnectAdminAuthToken(
                        builder -> builder.hostname(hostname).region(region).expiresIn(Duration.ofMinutes(30)).build());

                return this.getConnection(username, password);
            }

            @Override
            public Connection getConnection(final String username, final String password) throws SQLException {

                final Properties info = new Properties(props);
                info.setProperty("user", username);
                info.setProperty("password", password);
                return DriverManager.getConnection(url, info);
            }
        };

        final HikariDataSource hikariCp = new HikariDataSource();
        hikariCp.setDataSource(dataSource);
        hikariCp.setAutoCommit(false);
        hikariCp.setIdleTimeout(0L);

        return hikariCp;
    }
}
