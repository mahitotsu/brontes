package com.mahitotsu.brontes.api;

import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.cache.annotation.EnableCaching;

@SpringBootApplication
@EnableCaching
public class Main {
    
    public static void main(final String ...args) {
        new SpringApplicationBuilder(Main.class).web(WebApplicationType.REACTIVE).run(args);
    }
}