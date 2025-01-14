package com.mahitotsu.brontes.api;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import com.amazonaws.serverless.exceptions.ContainerInitializationException;
import com.amazonaws.serverless.proxy.model.AwsProxyResponse;
import com.amazonaws.serverless.proxy.model.HttpApiV2ProxyRequest;
import com.amazonaws.serverless.proxy.spring.SpringBootLambdaContainerHandler;
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestStreamHandler;

public class Handler implements RequestStreamHandler {

    private static SpringBootLambdaContainerHandler<HttpApiV2ProxyRequest, AwsProxyResponse> handler;
    static {
        try {
            handler = SpringBootLambdaContainerHandler.getHttpApiV2ProxyHandler(Main.class);
        } catch (ContainerInitializationException e) {
            e.printStackTrace();
            throw new RuntimeException("Could not initialize Spring Boot application", e);
        }
    }

    @Override
    public void handleRequest(final InputStream inputStream, final OutputStream outputStream, final Context context)
            throws IOException {
        Handler.handler.proxyStream(inputStream, outputStream, context);
    }
}
