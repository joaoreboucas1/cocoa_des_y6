# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if [ -z "${IGNORE_COSMOLIKE_DES_Y6_CODE}" ]; then

  if [ -z "${ROOTDIR}" ]; then
    source start_cocoa.sh || { pfail 'ROOTDIR'; return 1; }
  fi

  # Parenthesis = run in a subshell
  ( source "${ROOTDIR:?}/installation_scripts/flags_check.sh" ) || return 1;

  FOLDER="${DES_Y6_NAME:-"des_y6"}"
  PACKDIR="${ROOTDIR:?}/projects/${FOLDER:?}"

  export LD_LIBRARY_PATH="${PACKDIR:?}/interface":${LD_LIBRARY_PATH}

  export PYTHONPATH="${PACKDIR:?}/interface":${PYTHONPATH}

  if [ -n "${COSMOLIKE_DEBUG_MODE}" ]; then
      export SPDLOG_LEVEL=debug
  else
      export SPDLOG_LEVEL=info
  fi

  unset -v FOLDER PACKDIR

fi
