package com.mahitotsu.brontes.api.repository;

import java.io.IOException;
import java.nio.charset.Charset;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;

@Component
public class QueryLoader {

    @Autowired
    private ResourceLoader resourceLoader;

    @Cacheable(cacheNames = "NamedQueries", key = "#groupName + '/' + #queryName")
    public String loadNamedQuery(final String groupName, final String queryName) {

        final String path = "classpath:queries/" + groupName + "/" + queryName + ".sql";
        try {
            return this.resourceLoader.getResource(path)
                    .getContentAsString(Charset.defaultCharset());
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
