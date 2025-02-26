package com.mahitotsu.brontes.api.service;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import org.hibernate.exception.LockAcquisitionException;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;

@Target({ ElementType.METHOD, ElementType.TYPE })
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Retryable(retryFor = LockAcquisitionException.class, maxAttempts = 5, backoff = @Backoff(delay = 300, multiplier = 1.5, maxDelay = 5000, random = true))
public @interface DsqlRetry {

}
