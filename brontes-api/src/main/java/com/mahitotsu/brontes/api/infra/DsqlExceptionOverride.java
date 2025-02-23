package com.mahitotsu.brontes.api.infra;

import java.sql.SQLException;

import com.zaxxer.hikari.SQLExceptionOverride;

public class DsqlExceptionOverride implements SQLExceptionOverride {

        @java.lang.Override
        public Override adjudicate(final SQLException e) {
            final String sqlState = e.getSQLState();
            if ("0C000".equals(sqlState) || "0C001".equals(sqlState) || sqlState.matches("0A\\d{3}")) {
                return Override.DO_NOT_EVICT;
            }
            return Override.CONTINUE_EVICT;
        }
}
