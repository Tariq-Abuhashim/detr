FROM missionsystems/detr-devel-amd64:latest as intermediate

FROM ubuntu:20.04 

COPY --from=intermediate var/        var/
COPY --from=intermediate etc/        etc/
COPY --from=intermediate lib64/      lib64/
COPY --from=intermediate lib/        lib/  
COPY --from=intermediate bin/        bin/     
COPY --from=intermediate opt/        opt/   
COPY --from=intermediate usr/        usr/ 

WORKDIR /home/devops/hyperteaming
RUN mkdir -p /home/devops/src/hyperteaming
RUN chown -R devops:devops /home/devops
USER devops
