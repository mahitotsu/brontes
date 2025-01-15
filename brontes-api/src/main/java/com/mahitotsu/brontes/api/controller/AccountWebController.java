package com.mahitotsu.brontes.api.controller;

import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.mahitotsu.brontes.api.service.AccountService;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@RestController
@RequestMapping(path = "/api")
@Validated
public class AccountWebController {

    public static enum Operator {
        ADD, SUB,
    }

    @Data
    public static class CurrentValueResponse {
        @NotBlank
        private String accountId;
        private long currentValue;
    }

    @Data
    public static class CurrentValueRequest {
        @NotBlank
        private String accountId;
    }

    @Data
    public static class UpdateValueRequest {
        @NotBlank
        private String accountId;
        @NotNull
        private Operator operator;
        @Min(0)
        private long operand;
        private UUID requestId;
    }

    @Autowired
    private AccountService accountService;

    @PostMapping(path = "/get-value")
    public @Valid CurrentValueResponse getValue(@RequestBody @NotNull @Valid final CurrentValueRequest request) {

        final String accountId = request.getAccountId();
        final long currentValue = this.accountService.processGetCurrentRequest(accountId);

        final CurrentValueResponse response = new CurrentValueResponse();
        response.setAccountId(accountId);
        response.setCurrentValue(currentValue);

        return response;
    }

    @PostMapping(path = "/update-value")
    public @Valid CurrentValueResponse updateValue(@RequestBody @NotNull @Valid final UpdateValueRequest request)
            throws InterruptedException, ExecutionException, TimeoutException {

        final String accountId = request.getAccountId();
        final String operator = request.getOperator().name();
        final long operand = request.getOperand();

        UUID requestId = request.getRequestId();
        long currentValue;

        if (requestId == null) {
            requestId = this.accountService.queueUpdateRequest(accountId, operator, operand);
            currentValue = this.accountService.awaitUpdateResponse(accountId, requestId);
        } else {
            currentValue = this.accountService.processUpdateRequest(requestId, accountId, operator, operand);
        }

        final CurrentValueResponse response = new CurrentValueResponse();
        response.setAccountId(accountId);
        response.setCurrentValue(currentValue);

        return response;
    }
}
