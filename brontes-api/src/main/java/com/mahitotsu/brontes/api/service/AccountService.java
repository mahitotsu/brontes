package com.mahitotsu.brontes.api.service;

import java.util.Optional;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Service;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import software.amazon.awssdk.enhanced.dynamodb.DynamoDbEnhancedClient;
import software.amazon.awssdk.enhanced.dynamodb.DynamoDbTable;
import software.amazon.awssdk.enhanced.dynamodb.TableSchema;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbAttribute;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbBean;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbPartitionKey;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbSortKey;

@Service
public class AccountService {

    @Data
    @DynamoDbBean
    public static class HttpRequestTableSchema<RQ, RS> {
        private String group;
        private String requestId;
        private RQ request;
        private RS response;

        @DynamoDbPartitionKey
        @DynamoDbAttribute("group")
        public String getGroup() {
            return this.group;
        }

        @DynamoDbSortKey
        @DynamoDbAttribute("request-id")
        public String getRequestId() {
            return this.requestId;
        }

        @DynamoDbAttribute("request")
        public RQ getRequest() {
            return this.request;
        }

        @DynamoDbAttribute("response")
        public RS getResponse() {
            return this.response;
        }
    }

    @Data
    @DynamoDbBean
    public static class UpdateValueRequest {
        private String operator;
        private long operand;

        @DynamoDbAttribute("operator")
        public String getOperator() {
            return this.operator;
        }

        @DynamoDbAttribute("operand")
        public long getOperand() {
            return this.operand;
        }
    }

    @Data
    @DynamoDbBean
    public static class CurrentValueResponse {
        private long currentValue;

        @DynamoDbAttribute("current-value")
        public long getCurrentValue() {
            return this.currentValue;
        }
    }

    @Data
    @ToString(callSuper = true)
    @EqualsAndHashCode(callSuper = true)
    @DynamoDbBean
    public static class UpdateRequestItem extends HttpRequestTableSchema<UpdateValueRequest, CurrentValueResponse> {
    }

    public static class ItemNotFoundException extends RuntimeException {
    }

    @Value("${HTTP_REQUEST_TABLE_NAME}")
    private String httpRequestTableName;

    @Autowired
    private DynamoDbEnhancedClient dynamoDbClient;

    public long processGetCurrentRequest(final String accountId) {

        // query rdb
        final long currentValue = 0;

        // 
        return currentValue;
    }

    public UUID queueUpdateRequest(final String accountId, final String operator, final long operand) {

        final UpdateValueRequest request = new UpdateValueRequest();
        request.setOperator(operator);
        request.setOperand(operand);

        final UUID requestId = UUID.randomUUID();
        final UpdateRequestItem item = new UpdateRequestItem();
        item.setGroup(accountId);
        item.setRequestId(requestId.toString());
        item.setRequest(request);

        // queue request item
        final DynamoDbTable<UpdateRequestItem> table = this.dynamoDbClient.table(this.httpRequestTableName,
                TableSchema.fromBean(UpdateRequestItem.class));
        table.putItem(item);

        // return the assgined new request id
        return requestId;
    }

    @Retryable(include = ItemNotFoundException.class, maxAttempts = 10, backoff = @Backoff(delay = 100, multiplier = 1.2, random = true))
    public long awaitUpdateResponse(final String accountId, final UUID requestId) {

        // inquire the item for specified primary key
        final DynamoDbTable<UpdateRequestItem> table = this.dynamoDbClient.table(this.httpRequestTableName,
                TableSchema.fromBean(UpdateRequestItem.class));
        final Optional<UpdateRequestItem> item = Optional.ofNullable(table
                .getItem(r -> r.key(k -> k.partitionValue(accountId).sortValue(requestId.toString()))));

        // get the response,
        // or perform a retry if the response hasn't been provided yet.
        return item.map(i -> i.getResponse()).map(r -> r.getCurrentValue()).orElseThrow(ItemNotFoundException::new);
    }

    public Long processUpdateRequest(final UUID requestId, final String accountId, final String operator,
            final long operand) {

        // get the request to be processed
        final DynamoDbTable<UpdateRequestItem> table = this.dynamoDbClient.table(this.httpRequestTableName,
                TableSchema.fromBean(UpdateRequestItem.class));
        final UpdateRequestItem item = Optional.ofNullable(table
                .getItem(r -> r.key(k -> k.partitionValue(accountId).sortValue(requestId.toString()))))
                .orElseThrow(ItemNotFoundException::new);

        // update rdb
        final long currentValue = 0;

        // provide the response
        final CurrentValueResponse response = new CurrentValueResponse();
        response.setCurrentValue(currentValue);
        item.setResponse(response);
        table.putItem(item);

        // return the current vlaue
        return currentValue;
    }
}
