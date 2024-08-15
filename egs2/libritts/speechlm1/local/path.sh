if [ -z "${PYTHONPATH}" ]; then
  export PYTHONPATH=""
fi
export PYTHONPATH=${PYTHONPATH}:./encodec_16k_6kbps_multiDisc
