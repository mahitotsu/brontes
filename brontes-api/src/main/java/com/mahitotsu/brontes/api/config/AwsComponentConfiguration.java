package com.mahitotsu.brontes.api.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.enhanced.dynamodb.DynamoDbEnhancedClient;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;

@Configuration
public class AwsComponentConfiguration {

    @Bean
    public AwsCredentialsProvider awsCredentialsProvider() {
        return DefaultCredentialsProvider.create();
    }

    @Bean
    public DynamoDbEnhancedClient dynamoDbEnhancedClient(final AwsCredentialsProvider awsCredentialsProvider) {
        return DynamoDbEnhancedClient.builder()
                .dynamoDbClient(DynamoDbClient.builder().credentialsProvider(awsCredentialsProvider).build()).build();
    }
}
