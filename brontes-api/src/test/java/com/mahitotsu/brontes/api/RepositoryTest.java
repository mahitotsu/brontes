package com.mahitotsu.brontes.api;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.annotation.DirtiesContext.ClassMode;

import com.mahitotsu.brontes.api.config.DataConfig;

import io.r2dbc.spi.ConnectionFactory;

@SpringBootTest
@Import({ DataConfig.class })
@DirtiesContext(classMode = ClassMode.AFTER_CLASS)
public class RepositoryTest {

    @Autowired
    private ConnectionFactory connectionFactory;

    @Test
    public void testConnectionAvailable() {
        assertNotNull(this.connectionFactory);
    }
}
