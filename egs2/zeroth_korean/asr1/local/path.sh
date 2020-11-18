if ! command -v morfessor >/dev/null 2>&1; then
  echo "You appear to not have Morfessor installed, either on your path."
  echo "try: pip install morfessor"
  return 1
fi
if ! command -v flac >&/dev/null; then
   echo "Please install 'flac' on ALL worker nodes!"
   return 1
fi
