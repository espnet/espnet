# #!/usr/bin/env bash

# RED='\033[0;31m'
# GREEN='\033[0;32m'
# NC='\033[0m'

# check_command() {
#     local cmd=$1
#     local package=$2
#     if ! command -v "${cmd}" &> /dev/null; then
#         echo -e "${RED}Error: ${cmd} command not found.${NC}"
#         echo -e "Please install ${package}:"
#         echo "Ubuntu/Debian: sudo apt-get install ${package}"
#         return 1
#     else
#         echo -e "${GREEN}✓ ${cmd} is installed${NC}"
#         return 0
#     fi
# }

# echo "Checking required commands..."

# # Check commands and store results
# check_command "7z" "p7zip-full" || exit 1
# check_command "parallel" "parallel" || exit 1

# export PATH="$PATH"
# export LANG=en_US.UTF-8
# return 0
