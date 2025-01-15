package com.mahitotsu.brontes.api.controller;

import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;

import jakarta.validation.ConstraintViolationException;

@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler
    @ResponseBody
    public String handleException(final MethodArgumentNotValidException error) {
        return error.getMessage();
    }

    @ExceptionHandler
    @ResponseBody
    public String handleException(final ConstraintViolationException error) {
        return error.getMessage();
    }
}
