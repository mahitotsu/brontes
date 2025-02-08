package com.mahitotsu.brontes.api.config;

import java.lang.reflect.Method;
import java.time.Duration;
import java.util.Optional;
import java.util.function.Supplier;

import org.aopalliance.intercept.MethodInterceptor;
import org.springframework.aop.Advisor;
import org.springframework.aop.support.StaticMethodMatcherPointcutAdvisor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.boot.autoconfigure.r2dbc.ConnectionFactoryOptionsBuilderCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.springframework.context.annotation.Role;
import org.springframework.core.annotation.Order;
import org.springframework.core.io.ResourceLoader;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.lang.NonNull;
import org.springframework.r2dbc.connection.init.ConnectionFactoryInitializer;
import org.springframework.r2dbc.connection.init.DatabasePopulator;
import org.springframework.r2dbc.connection.init.ResourceDatabasePopulator;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import io.r2dbc.spi.ConnectionFactory;
import io.r2dbc.spi.ConnectionFactoryOptions;
import io.r2dbc.spi.Option;
import reactor.core.publisher.Mono;
import reactor.util.retry.RetrySpec;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
@Role(BeanDefinition.ROLE_INFRASTRUCTURE)
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
        final Supplier<CharSequence> tokenSupplier = () -> utilities
                .generateDbConnectAdminAuthToken(builder -> builder
                        .hostname(endpoint)
                        .expiresIn(Duration.ofSeconds(10))
                        .build());

        return builder -> {
            builder.option(Option.sensitiveValueOf(ConnectionFactoryOptions.PASSWORD.name()),
                    tokenSupplier);
        };
    }

    @Bean
    @Order(Integer.MAX_VALUE)
    @Role(BeanDefinition.ROLE_INFRASTRUCTURE)
    public Advisor retryAdvisor() {

        final MethodInterceptor retryInterceptor = (invocation) -> {

            final Object result = invocation.proceed();
            if (Mono.class.isInstance(result) == false) {
                return result;
            }

            final Mono<?> mono = Mono.class.cast(result);
            return mono.retryWhen(RetrySpec.backoff(5, Duration.ofMillis(500))
                    .filter(t -> CannotAcquireLockException.class.isInstance(t)));
        };

        return new StaticMethodMatcherPointcutAdvisor(retryInterceptor) {
            @Override
            public boolean matches(@NonNull final Method method, @NonNull final Class<?> targetClass) {
                return targetClass.isAnnotationPresent(Repository.class)
                        && Mono.class.isAssignableFrom(method.getReturnType())
                        && Optional.ofNullable(method.getAnnotation(Transactional.class))
                                .filter(tx -> tx.readOnly() == false).isPresent();
            }
        };
    }

    @Bean
    @Profile("drop-create")
    public ConnectionFactoryInitializer connectionFactoryInitializer(final ConnectionFactory connectionFactory) {

        final DatabasePopulator creator = new ResourceDatabasePopulator(
                this.resourceLoader.getResource("classpath:initdb/drop-create/schema-drop.sql"),
                this.resourceLoader.getResource("classpath:initdb/drop-create/schema-create.sql"));
        final DatabasePopulator cleaner = new ResourceDatabasePopulator(
                this.resourceLoader.getResource("classpath:initdb/drop-create/schema-drop.sql"));

        final ConnectionFactoryInitializer initializer = new ConnectionFactoryInitializer();
        initializer.setConnectionFactory(connectionFactory);
        initializer.setDatabasePopulator(creator);
        initializer.setDatabaseCleaner(cleaner);

        return initializer;
    }
}
