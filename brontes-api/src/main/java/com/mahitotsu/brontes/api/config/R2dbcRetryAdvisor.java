package com.mahitotsu.brontes.api.config;

import java.lang.reflect.Method;
import java.time.Duration;
import java.util.Optional;

import org.aopalliance.intercept.MethodInterceptor;
import org.aopalliance.intercept.MethodInvocation;
import org.springframework.aop.support.StaticMethodMatcherPointcutAdvisor;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.context.annotation.Role;
import org.springframework.core.annotation.Order;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.lang.NonNull;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import reactor.core.publisher.Mono;
import reactor.util.retry.RetrySpec;

@Component
@Role(BeanDefinition.ROLE_INFRASTRUCTURE)
@Order(Integer.MAX_VALUE)
public class R2dbcRetryAdvisor extends StaticMethodMatcherPointcutAdvisor {

    public R2dbcRetryAdvisor() {
        super(new MethodInterceptor() {
            @Override
            public Object invoke(MethodInvocation invocation) throws Throwable {
                final Object result = invocation.proceed();
                if (Mono.class.isInstance(result) == false) {
                    return result;
                }
                final Mono<?> mono = Mono.class.cast(result);
                return mono.retryWhen(RetrySpec.backoff(5, Duration.ofMillis(500))
                        .filter(t -> CannotAcquireLockException.class.isInstance(t)));
            }
        });
    }

    @Override
    public boolean matches(@NonNull final Method method, @NonNull final Class<?> targetClass) {
        return targetClass.isAnnotationPresent(Repository.class)
                && Mono.class.isAssignableFrom(method.getReturnType())
                && Optional.ofNullable(method.getAnnotation(Transactional.class))
                        .filter(tx -> tx.readOnly() == false).isPresent();
    }
}
