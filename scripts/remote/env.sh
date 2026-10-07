# Source on a server: puts <repo>/../SOPT_ENV first on PATH and exports SOPT_DATA for that host.
_SOPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PATH="$(dirname "$_SOPT_ROOT")/SOPT_ENV/bin:$PATH"
case "$(hostname)" in
  nyuair)   export SOPT_DATA=$HOME/langtian/SOPT_DATA ;;
  chatsign) export SOPT_DATA=$HOME/Desktop/langtian/SOPT_DATA ;;
  *)        export SOPT_DATA=${SOPT_DATA:-/scratch/$USER/SOPT_DATA} ;;
esac
