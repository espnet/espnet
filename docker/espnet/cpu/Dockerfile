# This will need to be changed when the official release is done
FROM fhrozen/dev_espnet:runtime
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

# Install espnet cpu
WORKDIR /espnet/tools

RUN make KALDI=/kaldi all 

WORKDIR /