FROM public.ecr.aws/lambda/java:21

COPY target/classes ${LAMBDA_TASK_ROOT}
COPY target/dependency/* ${LAMBDA_TASK_ROOT}/lib/
RUN rm ${LAMBDA_TASK_ROOT}/lib/commons-logging-*.jar

CMD ["com.mahitotsu.brontes.api.Handler::handleRequest"]