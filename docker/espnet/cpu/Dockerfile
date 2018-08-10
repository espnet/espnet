# This will need to be changed when the official release is done
FROM fhrozen/dev_espnet:runtime

# Install espnet cpu
RUN mkdir -p /espnet/tools 

WORKDIR /espnet/tools
RUN wget https://raw.githubusercontent.com/espnet/espnet/master/tools/Makefile && \
    wget https://raw.githubusercontent.com/espnet/espnet/master/tools/requirements.txt && \
    wwget https://github.com/espnet/espnet/blob/master/tools/test_requirements.txt

RUN make KALDI=/kaldi all 

WORKDIR /